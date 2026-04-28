#[macro_export]
macro_rules! dispatch_cpal_sample_format {
    ($sample_format:expr, $builder:ident $(, $arg:expr )* ; unsupported => $unsupported:expr $(,)?) => {
        match $sample_format {
            cpal::SampleFormat::F32 => $builder::<f32>($($arg),*),
            cpal::SampleFormat::I16 => $builder::<i16>($($arg),*),
            cpal::SampleFormat::U16 => $builder::<u16>($($arg),*),
            cpal::SampleFormat::I8 => $builder::<i8>($($arg),*),
            cpal::SampleFormat::U8 => $builder::<u8>($($arg),*),
            cpal::SampleFormat::I32 => $builder::<i32>($($arg),*),
            cpal::SampleFormat::U32 => $builder::<u32>($($arg),*),
            cpal::SampleFormat::I64 => $builder::<i64>($($arg),*),
            cpal::SampleFormat::U64 => $builder::<u64>($($arg),*),
            cpal::SampleFormat::F64 => $builder::<f64>($($arg),*),
            _ => $unsupported,
        }
    };
}
