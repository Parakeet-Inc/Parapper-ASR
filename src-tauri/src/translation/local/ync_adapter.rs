use tiny_http::Method;

pub(super) const PATH: &str = "/";

pub(super) fn accepts(method: &Method, path: &str) -> bool {
    method == &Method::Post && path == PATH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ync_adapter_accepts_only_plugin_root_post() {
        assert!(accepts(&Method::Post, "/"));
        assert!(!accepts(&Method::Post, "/api/input"));
        assert!(!accepts(&Method::Post, "/v1/chat/completions"));
        assert!(!accepts(&Method::Get, "/"));
    }
}
