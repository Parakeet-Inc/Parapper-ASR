use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub struct TextInputPayload<'a> {
    pub text: &'a str,
    pub is_final: bool,
    pub text_id: &'a str,
}

pub trait TextTransport: Send {
    fn send_text(&mut self, payload: TextInputPayload<'_>) -> Result<()>;
}
