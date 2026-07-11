use tiny_http::Method;

pub(super) const PATH: &str = "/v1/chat/completions";

pub(super) fn accepts(method: &Method, path: &str) -> bool {
    method == &Method::Post && path == PATH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_adapter_accepts_only_chat_completions_post() {
        assert!(accepts(&Method::Post, "/v1/chat/completions"));
        assert!(!accepts(&Method::Post, "/"));
        assert!(!accepts(&Method::Post, "/api/input"));
        assert!(!accepts(&Method::Get, "/v1/chat/completions"));
    }
}
