use std::{
    io::{ErrorKind, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

type RequestHandler = dyn Fn(&str, usize) -> String + Send + Sync;

pub(crate) struct MockHttpServer {
    port: u16,
    requests: mpsc::Receiver<String>,
    handle: JoinHandle<()>,
}

#[derive(Debug)]
pub(crate) struct TimedHttpRequest {
    pub(crate) received_at: Instant,
    pub(crate) raw: String,
}

pub(crate) struct TimedMockHttpServer {
    port: u16,
    requests: mpsc::Receiver<TimedHttpRequest>,
    handle: JoinHandle<()>,
}

impl MockHttpServer {
    pub(crate) fn start<H>(request_count: usize, handler: H) -> Self
    where
        H: Fn(&str, usize) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_sender, requests) = mpsc::channel();
        let handler: Arc<RequestHandler> = Arc::new(handler);
        let handle = thread::spawn(move || {
            let mut workers = Vec::new();
            for index in 0..request_count {
                let (stream, _) = listener.accept().unwrap();
                let request_sender = request_sender.clone();
                let handler = handler.clone();
                workers.push(thread::spawn(move || {
                    handle_request(stream, index, &request_sender, &handler);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            port,
            requests,
            handle,
        }
    }

    pub(crate) fn start_until_idle<H>(idle_timeout: Duration, handler: H) -> Self
    where
        H: Fn(&str, usize) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_sender, requests) = mpsc::channel();
        let handler: Arc<RequestHandler> = Arc::new(handler);
        let handle = thread::spawn(move || {
            let mut workers = Vec::new();
            let mut index = 0;
            let mut deadline = None;
            loop {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        let request_sender = request_sender.clone();
                        let handler = handler.clone();
                        workers.push(thread::spawn(move || {
                            handle_request(stream, index, &request_sender, &handler);
                        }));
                        index += 1;
                        deadline = Some(Instant::now() + idle_timeout);
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(err) => panic!("mock server accept failed: {err}"),
                }
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            port,
            requests,
            handle,
        }
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn recv_request(&self) -> String {
        self.requests.recv_timeout(Duration::from_secs(1)).unwrap()
    }

    pub(crate) fn try_recv_request(&self, timeout: Duration) -> Option<String> {
        self.requests.recv_timeout(timeout).ok()
    }

    pub(crate) fn join(self) {
        self.handle.join().unwrap();
    }
}

impl TimedMockHttpServer {
    pub(crate) fn start<H>(request_count: usize, handler: H) -> Self
    where
        H: Fn(&str, usize) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_sender, requests) = mpsc::channel();
        let handler: Arc<RequestHandler> = Arc::new(handler);
        let handle = thread::spawn(move || {
            let mut workers = Vec::new();
            for index in 0..request_count {
                let (stream, _) = listener.accept().unwrap();
                let request_sender = request_sender.clone();
                let handler = handler.clone();
                workers.push(thread::spawn(move || {
                    handle_timed_request(stream, index, &request_sender, &handler);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            port,
            requests,
            handle,
        }
    }

    pub(crate) fn start_until_idle<H>(idle_timeout: Duration, handler: H) -> Self
    where
        H: Fn(&str, usize) -> String + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_sender, requests) = mpsc::channel();
        let handler: Arc<RequestHandler> = Arc::new(handler);
        let handle = thread::spawn(move || {
            let mut workers = Vec::new();
            let mut index = 0;
            let mut deadline = None;
            loop {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        let request_sender = request_sender.clone();
                        let handler = handler.clone();
                        workers.push(thread::spawn(move || {
                            handle_timed_request(stream, index, &request_sender, &handler);
                        }));
                        index += 1;
                        deadline = Some(Instant::now() + idle_timeout);
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(err) => panic!("timed mock server accept failed: {err}"),
                }
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            port,
            requests,
            handle,
        }
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn recv_request(&self) -> TimedHttpRequest {
        self.requests.recv_timeout(Duration::from_secs(1)).unwrap()
    }

    pub(crate) fn try_recv_request(&self, timeout: Duration) -> Option<TimedHttpRequest> {
        self.requests.recv_timeout(timeout).ok()
    }

    pub(crate) fn join(self) {
        self.handle.join().unwrap();
    }
}

fn handle_request(
    mut stream: TcpStream,
    index: usize,
    request_sender: &mpsc::Sender<String>,
    handler: &Arc<RequestHandler>,
) {
    stream.set_nonblocking(false).unwrap();
    let mut buffer = [0_u8; 4096];
    let len = stream.read(&mut buffer).unwrap();
    let request = String::from_utf8_lossy(&buffer[..len]).to_string();
    request_sender.send(request.clone()).unwrap();
    let response = handler(&request, index);
    stream.write_all(response.as_bytes()).unwrap();
}

fn handle_timed_request(
    mut stream: TcpStream,
    index: usize,
    request_sender: &mpsc::Sender<TimedHttpRequest>,
    handler: &Arc<RequestHandler>,
) {
    stream.set_nonblocking(false).unwrap();
    let mut buffer = [0_u8; 4096];
    let len = stream.read(&mut buffer).unwrap();
    let request = String::from_utf8_lossy(&buffer[..len]).to_string();
    request_sender
        .send(TimedHttpRequest {
            received_at: Instant::now(),
            raw: request.clone(),
        })
        .unwrap();
    let response = handler(&request, index);
    stream.write_all(response.as_bytes()).unwrap();
}

pub(crate) fn json_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

pub(crate) fn text_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

pub(crate) fn request_id_from_plugin_request(request: &str) -> &str {
    let id_start = request
        .find(r#""id":""#)
        .map(|index| index + r#""id":""#.len())
        .expect("plugin request should contain id");
    let id_end = request[id_start..]
        .find('"')
        .map(|index| id_start + index)
        .expect("plugin request id should be terminated");
    &request[id_start..id_end]
}
