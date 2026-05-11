use anyhow::Result;

pub trait TextTransport: Send {
    fn send_text(&mut self, text: &str) -> Result<()>;
}
