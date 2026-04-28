//! `VRChat` のマイクミュート状態を、`OSCQuery` 経由で読み取るためのモジュールです。
//!
//! `Parapper` の目的は音声認識結果をゆかこねNEOへ渡すことなので、`VRChat` 側の
//! ミュート状態は「送信直前の判断材料」として扱います。発話が ASR worker に渡された
//! タイミングで `/avatar/parameters/MuteSelf` の確認を非同期に開始し、認識結果を
//! 送信する直前にその結果を参照します。`VRChat` 側がミュート中なら、認識結果の記録は
//! 残しつつ、ゆかこねNEOへの送信だけをスキップします。
//!
//! `OSCQuery` の探索や HTTP 読み取りは、`VRChat` の起動タイミングや一時的な状態に
//! 左右されます。そのため、問い合わせ結果は送信可否の補助情報として扱います。取得できない
//! 場合は「ミュートではない」ものとして扱い、ASR パイプラインとゆかこねNEOへの送信を
//! 継続します。
//!
//! `OSCQuery` のエンドポイントは、`_oscjson._tcp.local.` の mDNS browse で発見します。
//! これは `OSCQuery` が想定している発見方法で、見つかった `VRChat-Client` サービスの
//! HTTP エンドポイントだけを `MuteSelf` 確認先として使います。

use std::{
    net::{IpAddr, SocketAddr},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use mdns_sd::{ResolvedService, ServiceDaemon, ServiceEvent};
use serde_json::Value;

const OSCQUERY_SERVICE_TYPE: &str = "_oscjson._tcp.local.";
const VRCHAT_SERVICE_NAME_PREFIX: &str = "VRChat-Client";
const MUTE_SELF_PATH: &str = "/avatar/parameters/MuteSelf";
const HTTP_TIMEOUT: Duration = Duration::from_millis(500);
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);
const DISCOVERY_FAILURE_COOLDOWN: Duration = Duration::from_secs(10);
static OSCQUERY_ADDR_CACHE: OnceLock<Mutex<Option<SocketAddr>>> = OnceLock::new();
static OSCQUERY_DISCOVERY_COOLDOWN: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

fn discover_vrchat_oscquery_addr() -> Option<SocketAddr> {
    if discovery_is_in_cooldown() {
        return None;
    }

    let discovered_addr = discover_vrchat_oscquery_service();
    if discovered_addr.is_none() {
        start_discovery_cooldown();
    }
    discovered_addr
}

pub fn query_current_mute_state() -> Result<bool> {
    if let Some(addr) = cached_oscquery_addr() {
        match fetch_oscquery_mute_state(addr) {
            Ok(is_muted) => return Ok(is_muted),
            Err(_) => {
                set_cached_oscquery_addr(None);
            }
        }
    }

    let addr =
        discover_vrchat_oscquery_addr().ok_or_else(|| anyhow!("VRChat OSCQuery not found"))?;
    set_cached_oscquery_addr(Some(addr));
    fetch_oscquery_mute_state(addr)
}

fn discover_vrchat_oscquery_service() -> Option<SocketAddr> {
    let mdns = ServiceDaemon::new().ok()?;
    let receiver = mdns.browse(OSCQUERY_SERVICE_TYPE).ok()?;
    let deadline = Instant::now() + DISCOVERY_TIMEOUT;
    let mut discovered_addr = None;

    while Instant::now() < deadline {
        let timeout = deadline.saturating_duration_since(Instant::now());
        match receiver.recv_timeout(timeout) {
            Ok(ServiceEvent::ServiceResolved(info)) if is_vrchat_oscquery_service(&info) => {
                discovered_addr = service_socket_addr(&info);
                if discovered_addr.is_some() {
                    break;
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    let _ = mdns.stop_browse(OSCQUERY_SERVICE_TYPE);
    let _ = mdns.shutdown();
    discovered_addr
}

fn is_vrchat_oscquery_service(info: &ResolvedService) -> bool {
    info.get_fullname().starts_with(VRCHAT_SERVICE_NAME_PREFIX)
}

fn service_socket_addr(info: &ResolvedService) -> Option<SocketAddr> {
    let port = info.get_port();
    preferred_ip_addr(info)
        .or_else(|| {
            info.get_addresses()
                .iter()
                .next()
                .map(mdns_sd::ScopedIp::to_ip_addr)
        })
        .map(|ip| SocketAddr::new(ip, port))
}

fn preferred_ip_addr(info: &ResolvedService) -> Option<IpAddr> {
    info.get_addresses()
        .iter()
        .find(|ip| ip.is_loopback())
        .map(mdns_sd::ScopedIp::to_ip_addr)
        .or_else(|| {
            info.get_addresses()
                .iter()
                .find(|ip| ip.is_ipv4())
                .map(mdns_sd::ScopedIp::to_ip_addr)
        })
}

fn discovery_is_in_cooldown() -> bool {
    OSCQUERY_DISCOVERY_COOLDOWN
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|cooldown_until| *cooldown_until)
        .is_some_and(|cooldown_until| Instant::now() < cooldown_until)
}

fn start_discovery_cooldown() {
    if let Ok(mut cooldown_until) = OSCQUERY_DISCOVERY_COOLDOWN
        .get_or_init(|| Mutex::new(None))
        .lock()
    {
        *cooldown_until = Some(Instant::now() + DISCOVERY_FAILURE_COOLDOWN);
    }
}

fn cached_oscquery_addr() -> Option<SocketAddr> {
    OSCQUERY_ADDR_CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|addr| *addr)
}

fn set_cached_oscquery_addr(addr: Option<SocketAddr>) {
    if let Ok(mut cached_addr) = OSCQUERY_ADDR_CACHE.get_or_init(|| Mutex::new(None)).lock() {
        *cached_addr = addr;
    }
}

fn fetch_oscquery_mute_state(addr: SocketAddr) -> Result<bool> {
    match get_oscquery_response(addr, &format!("{MUTE_SELF_PATH}?VALUE"))
        .and_then(|body| parse_oscquery_mute_value(&body).ok_or_else(|| anyhow!("VALUE missing")))
    {
        Ok(is_muted) => Ok(is_muted),
        Err(value_err) => {
            let node_response = get_oscquery_response(addr, MUTE_SELF_PATH)
                .with_context(|| format!("Failed to read OSCQuery VALUE path: {value_err}"))?;
            parse_oscquery_mute_value(&node_response)
                .ok_or_else(|| anyhow!("OSCQuery response does not contain MuteSelf VALUE"))
        }
    }
}

fn get_oscquery_response(addr: SocketAddr, path: &str) -> Result<String> {
    let endpoint = addr.to_string();
    let url = format!("http://{endpoint}{path}");
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("Failed to build OSCQuery HTTP client")?;
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to read VRChat OSCQuery: {endpoint}"))?;
    if !response.status().is_success() {
        return Err(anyhow!("OSCQuery returned an error: {}", response.status()));
    }
    response
        .text()
        .context("OSCQuery response body is not valid text")
}

fn parse_oscquery_mute_value(body: &str) -> Option<bool> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    value
        .get("VALUE")
        .or_else(|| value.get("value"))
        .and_then(parse_boolish_value)
        .or_else(|| parse_boolish_value(&value))
}

fn parse_boolish_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Number(value) => value.as_i64().map(|value| value != 0),
        Value::String(value) => match value.as_str() {
            "true" | "TRUE" | "True" | "1" => Some(true),
            "false" | "FALSE" | "False" | "0" => Some(false),
            _ => None,
        },
        Value::Array(values) => values.first().and_then(parse_boolish_value),
        Value::Object(values) => values
            .get("VALUE")
            .or_else(|| values.get("value"))
            .and_then(parse_boolish_value),
        Value::Null => None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_oscquery_mute_value_reads_value_array() {
        assert_eq!(
            super::parse_oscquery_mute_value(r#"{"VALUE":[true]}"#),
            Some(true)
        );
    }

    #[test]
    fn parse_oscquery_mute_value_reads_numeric_value() {
        assert_eq!(
            super::parse_oscquery_mute_value(r#"{"VALUE":[0]}"#),
            Some(false)
        );
    }

    #[test]
    fn parse_oscquery_mute_value_reads_string_value() {
        assert_eq!(
            super::parse_oscquery_mute_value(r#"{"VALUE":"false"}"#),
            Some(false)
        );
    }
}
