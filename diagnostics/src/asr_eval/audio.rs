use std::error::Error;
use std::fmt;

const CANONICAL_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalWav {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalWavError {
    InvalidContainer,
    MissingFormatChunk,
    MissingDataChunk,
    TruncatedChunk {
        chunk: String,
    },
    UnsupportedFormat {
        audio_format: u16,
        sample_rate_hz: u32,
        channels: u16,
        bits_per_sample: u16,
    },
    InvalidDataLength {
        bytes: usize,
    },
}

/// Decodes the canonical PCM contract used by offline ASR evaluation.
///
/// No resampling, channel mixing, padding, or amplitude normalization occurs;
/// a non-canonical file fails before reaching inference.
///
/// # Errors
///
/// Returns a typed error for a malformed RIFF container, truncated or missing
/// chunks, or any format other than mono 16 kHz signed PCM16.
pub fn decode_canonical_pcm16_wav(bytes: &[u8]) -> Result<CanonicalWav, CanonicalWavError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(CanonicalWavError::InvalidContainer);
    }
    let riff_size = read_u32(bytes, 4).ok_or(CanonicalWavError::InvalidContainer)? as usize;
    let riff_end = riff_size
        .checked_add(8)
        .ok_or(CanonicalWavError::InvalidContainer)?;
    if riff_end > bytes.len() {
        return Err(CanonicalWavError::TruncatedChunk {
            chunk: "RIFF".to_owned(),
        });
    }

    let mut format = None;
    let mut data = None;
    let mut cursor = 12;
    while cursor < riff_end {
        if cursor + 8 > riff_end {
            return Err(CanonicalWavError::TruncatedChunk {
                chunk: "chunk header".to_owned(),
            });
        }
        let id = &bytes[cursor..cursor + 4];
        let size = read_u32(bytes, cursor + 4).ok_or_else(|| CanonicalWavError::TruncatedChunk {
            chunk: "chunk header".to_owned(),
        })? as usize;
        let payload_start = cursor + 8;
        let payload_end =
            payload_start
                .checked_add(size)
                .ok_or_else(|| CanonicalWavError::TruncatedChunk {
                    chunk: chunk_name(id),
                })?;
        if payload_end > riff_end {
            return Err(CanonicalWavError::TruncatedChunk {
                chunk: chunk_name(id),
            });
        }

        match id {
            b"fmt " => {
                if size < 16 {
                    return Err(CanonicalWavError::TruncatedChunk {
                        chunk: "fmt ".to_owned(),
                    });
                }
                format = Some(read_format(bytes, payload_start).ok_or_else(|| {
                    CanonicalWavError::TruncatedChunk {
                        chunk: "fmt ".to_owned(),
                    }
                })?);
            }
            b"data" => data = Some(&bytes[payload_start..payload_end]),
            _ => {}
        }

        cursor = payload_end + (size & 1);
    }

    let format = format.ok_or(CanonicalWavError::MissingFormatChunk)?;
    if format.audio_format != 1
        || format.sample_rate_hz != CANONICAL_SAMPLE_RATE_HZ
        || format.channels != 1
        || format.bits_per_sample != 16
    {
        return Err(CanonicalWavError::UnsupportedFormat {
            audio_format: format.audio_format,
            sample_rate_hz: format.sample_rate_hz,
            channels: format.channels,
            bits_per_sample: format.bits_per_sample,
        });
    }
    let data = data.ok_or(CanonicalWavError::MissingDataChunk)?;
    if data.len() % 2 != 0 {
        return Err(CanonicalWavError::InvalidDataLength { bytes: data.len() });
    }
    let samples = data
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
        .collect();
    Ok(CanonicalWav {
        sample_rate_hz: format.sample_rate_hz,
        samples,
    })
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let pair = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([pair[0], pair[1]]))
}

fn read_format(bytes: &[u8], offset: usize) -> Option<WavFormat> {
    Some(WavFormat {
        audio_format: read_u16(bytes, offset)?,
        channels: read_u16(bytes, offset + 2)?,
        sample_rate_hz: read_u32(bytes, offset + 4)?,
        bits_per_sample: read_u16(bytes, offset + 14)?,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let quartet = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([
        quartet[0], quartet[1], quartet[2], quartet[3],
    ]))
}

fn chunk_name(id: &[u8]) -> String {
    String::from_utf8_lossy(id).into_owned()
}

impl fmt::Display for CanonicalWavError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContainer => write!(formatter, "audio is not a RIFF/WAVE container"),
            Self::MissingFormatChunk => write!(formatter, "WAV fmt chunk is missing"),
            Self::MissingDataChunk => write!(formatter, "WAV data chunk is missing"),
            Self::TruncatedChunk { chunk } => write!(formatter, "WAV {chunk} chunk is truncated"),
            Self::UnsupportedFormat {
                audio_format,
                sample_rate_hz,
                channels,
                bits_per_sample,
            } => write!(
                formatter,
                "unsupported WAV format {audio_format}/{sample_rate_hz}Hz/{channels}ch/{bits_per_sample}bit; expected PCM/16000Hz/1ch/16bit"
            ),
            Self::InvalidDataLength { bytes } => {
                write!(formatter, "PCM16 WAV data has an odd byte length: {bytes}")
            }
        }
    }
}

impl Error for CanonicalWavError {}

#[cfg(test)]
mod tests {
    use super::{CanonicalWavError, decode_canonical_pcm16_wav};

    fn pcm16_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(samples.len() * 2).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(channels) * 2).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn canonical_pcm16_wav_decodes_each_sample_without_resampling_or_padding() {
        let wav =
            decode_canonical_pcm16_wav(&pcm16_wav(16_000, 1, &[i16::MIN, -1, 0, 1, i16::MAX]))
                .unwrap();

        assert_eq!(wav.sample_rate_hz, 16_000);
        assert_eq!(wav.samples.len(), 5);
        assert!((wav.samples[0] - -1.0).abs() <= f32::EPSILON);
        assert!(wav.samples[2].abs() <= f32::EPSILON);
        assert!((wav.samples[4] - f32::from(i16::MAX) / 32768.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn noncanonical_rate_channels_and_sample_format_are_rejected() {
        assert!(matches!(
            decode_canonical_pcm16_wav(&pcm16_wav(48_000, 1, &[0])),
            Err(CanonicalWavError::UnsupportedFormat {
                sample_rate_hz: 48_000,
                ..
            })
        ));
        assert!(matches!(
            decode_canonical_pcm16_wav(&pcm16_wav(16_000, 2, &[0, 0])),
            Err(CanonicalWavError::UnsupportedFormat { channels: 2, .. })
        ));

        let mut float_wav = pcm16_wav(16_000, 1, &[0]);
        float_wav[20..22].copy_from_slice(&3_u16.to_le_bytes());
        assert!(matches!(
            decode_canonical_pcm16_wav(&float_wav),
            Err(CanonicalWavError::UnsupportedFormat {
                audio_format: 3,
                ..
            })
        ));
    }

    #[test]
    fn truncated_or_missing_data_chunks_fail_before_inference() {
        let mut truncated = pcm16_wav(16_000, 1, &[1, 2]);
        truncated.pop();
        assert!(matches!(
            decode_canonical_pcm16_wav(&truncated),
            Err(CanonicalWavError::TruncatedChunk { .. })
        ));
        assert!(matches!(
            decode_canonical_pcm16_wav(b"not a wave"),
            Err(CanonicalWavError::InvalidContainer)
        ));
    }
}
