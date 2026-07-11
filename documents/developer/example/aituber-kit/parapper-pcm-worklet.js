class ParapperPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.phase = 0
    this.sum = 0
    this.count = 0
    this.frame = new Int16Array(512)
    this.frameOffset = 0
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    if (!channel) return true

    for (const sample of channel) {
      this.sum += sample
      this.count += 1
      this.phase += 16000

      if (this.phase < sampleRate) continue

      this.phase -= sampleRate
      const averaged = Math.max(-1, Math.min(1, this.sum / this.count))
      this.frame[this.frameOffset] =
        averaged < 0 ? averaged * 0x8000 : averaged * 0x7fff
      this.frameOffset += 1
      this.sum = 0
      this.count = 0

      if (this.frameOffset === this.frame.length) {
        const frame = this.frame
        this.port.postMessage(frame.buffer, [frame.buffer])
        this.frame = new Int16Array(512)
        this.frameOffset = 0
      }
    }

    return true
  }
}

registerProcessor('parapper-pcm-processor', ParapperPcmProcessor)
