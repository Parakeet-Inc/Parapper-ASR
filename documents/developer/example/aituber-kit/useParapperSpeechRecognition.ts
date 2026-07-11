import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

declare const process: {
  env: { NEXT_PUBLIC_PARAPPER_URL?: string }
}

const PARAPPER_URL =
  process.env.NEXT_PUBLIC_PARAPPER_URL ?? 'ws://127.0.0.1:18082/ws/recognition'

type ParapperMessage = {
  type: string
  session_id?: string
  text?: string
  turn_session_id?: number
  turn_id?: number
  code?: string
  message?: string
}

export function useParapperSpeechRecognition(
  onChatProcessStart: (text: string) => void
) {
  const [userMessage, setUserMessage] = useState('')
  const [isListening, setIsListening] = useState(false)
  const websocketRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const stopTimeoutRef = useRef<number | null>(null)
  const stopSentRef = useRef(false)
  const deliveredFinalsRef = useRef(new Set<string>())

  const stopAudio = useCallback(async () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    const context = audioContextRef.current
    audioContextRef.current = null
    if (context && context.state !== 'closed') await context.close()
    setIsListening(false)
  }, [])

  const finishListening = useCallback(async () => {
    await stopAudio()
    const websocket = websocketRef.current
    const sessionId = sessionIdRef.current
    if (
      !websocket ||
      !sessionId ||
      websocket.readyState !== WebSocket.OPEN ||
      stopSentRef.current
    ) {
      return
    }

    stopSentRef.current = true
    websocket.send(
      JSON.stringify({
        version: 1,
        type: 'session.stop',
        session_id: sessionId,
      })
    )
    if (stopTimeoutRef.current !== null) clearTimeout(stopTimeoutRef.current)
    stopTimeoutRef.current = window.setTimeout(() => websocket.close(), 5000)
  }, [stopAudio])

  const cancelListening = useCallback(async () => {
    await stopAudio()

    const websocket = websocketRef.current
    const sessionId = sessionIdRef.current
    websocketRef.current = null
    sessionIdRef.current = null
    stopSentRef.current = false
    deliveredFinalsRef.current.clear()

    if (websocket?.readyState === WebSocket.OPEN && sessionId) {
      websocket.send(
        JSON.stringify({
          version: 1,
          type: 'session.cancel',
          session_id: sessionId,
        })
      )
    }
    websocket?.close()
  }, [stopAudio])

  const startListening = useCallback(async () => {
    if (websocketRef.current || isListening) return

    const sessionId = `aituber-${crypto.randomUUID()}`
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true },
    })
    const websocket = new WebSocket(PARAPPER_URL)
    websocketRef.current = websocket
    sessionIdRef.current = sessionId
    streamRef.current = stream
    stopSentRef.current = false
    deliveredFinalsRef.current.clear()

    websocket.onclose = () => {
      if (stopTimeoutRef.current !== null) {
        clearTimeout(stopTimeoutRef.current)
        stopTimeoutRef.current = null
      }
      if (websocketRef.current === websocket) {
        websocketRef.current = null
        sessionIdRef.current = null
        stopSentRef.current = false
        deliveredFinalsRef.current.clear()
      }
      void stopAudio()
    }

    try {
      await new Promise<void>((resolve, reject) => {
        websocket.onopen = () => resolve()
        websocket.onerror = () =>
          reject(new Error('Parapper connection failed'))
      })

      const ready = new Promise<void>((resolve, reject) => {
        const rejectOnClose = () =>
          reject(new Error('Parapper closed before session.ready'))
        websocket.addEventListener('close', rejectOnClose, { once: true })
        websocket.onmessage = (event) => {
          if (typeof event.data !== 'string') return
          const message = JSON.parse(event.data) as ParapperMessage

          if (message.type === 'session.ready') {
            websocket.removeEventListener('close', rejectOnClose)
            resolve()
          } else if (message.type === 'turn.partial') {
            setUserMessage(message.text ?? '')
          } else if (message.type === 'turn.final') {
            const finalKey =
              message.turn_session_id !== undefined &&
              message.turn_id !== undefined
                ? `${message.turn_session_id}:${message.turn_id}`
                : null
            if (finalKey && deliveredFinalsRef.current.has(finalKey)) return
            if (finalKey) deliveredFinalsRef.current.add(finalKey)

            const text = message.text?.trim() ?? ''
            setUserMessage(text)
            if (text) onChatProcessStart(text)
          } else if (message.type === 'session.done') {
            if (stopTimeoutRef.current !== null) {
              clearTimeout(stopTimeoutRef.current)
              stopTimeoutRef.current = null
            }
            websocket.close()
          } else if (message.type === 'error') {
            websocket.removeEventListener('close', rejectOnClose)
            reject(
              new Error(
                `Parapper ${message.code ?? 'error'}: ${message.message ?? ''}`
              )
            )
            websocket.close()
          }
        }
      })

      websocket.send(
        JSON.stringify({
          version: 1,
          type: 'session.start',
          session_id: sessionId,
          audio: {
            encoding: 'pcm_s16le',
            sample_rate: 16000,
            channels: 1,
          },
        })
      )
      await ready

      const audioContext = new AudioContext()
      if (audioContext.sampleRate < 16000) {
        throw new Error(
          `Unsupported browser sample rate: ${audioContext.sampleRate}`
        )
      }
      audioContextRef.current = audioContext
      await audioContext.audioWorklet.addModule('/parapper-pcm-worklet.js')

      const source = audioContext.createMediaStreamSource(stream)
      const worklet = new AudioWorkletNode(
        audioContext,
        'parapper-pcm-processor'
      )
      const silentOutput = audioContext.createGain()
      silentOutput.gain.value = 0
      worklet.port.onmessage = ({ data }: MessageEvent<ArrayBuffer>) => {
        if (websocket.readyState === WebSocket.OPEN) websocket.send(data)
      }
      source
        .connect(worklet)
        .connect(silentOutput)
        .connect(audioContext.destination)
      if (audioContext.state === 'suspended') await audioContext.resume()
      setIsListening(true)
    } catch (error) {
      await stopAudio()
      websocket.close()
      websocketRef.current = null
      sessionIdRef.current = null
      throw error
    }
  }, [isListening, onChatProcessStart, stopAudio])

  const toggleListening = useCallback(() => {
    if (isListening) void finishListening()
    else void startListening()
  }, [finishListening, isListening, startListening])

  const handleInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setUserMessage(event.target.value)
    },
    []
  )

  const handleSendMessage = useCallback(async () => {
    const text = userMessage.trim()
    if (!text) return
    onChatProcessStart(text)
    setUserMessage('')
  }, [onChatProcessStart, userMessage])

  useEffect(
    () => () => {
      const websocket = websocketRef.current
      const sessionId = sessionIdRef.current
      if (
        websocket?.readyState === WebSocket.OPEN &&
        sessionId &&
        !stopSentRef.current
      ) {
        websocket.send(
          JSON.stringify({
            version: 1,
            type: 'session.cancel',
            session_id: sessionId,
          })
        )
      }
      websocket?.close()
      void stopAudio()
    },
    [stopAudio]
  )

  return useMemo(
    () => ({
      userMessage,
      isListening,
      silenceTimeoutRemaining: 0,
      handleInputChange,
      handleSendMessage,
      toggleListening,
      startListening,
      stopListening: cancelListening,
      checkRecognitionActive: () =>
        Boolean(
          websocketRef.current?.readyState === WebSocket.OPEN &&
          audioContextRef.current?.state === 'running'
        ),
    }),
    [
      handleInputChange,
      handleSendMessage,
      cancelListening,
      isListening,
      startListening,
      toggleListening,
      userMessage,
    ]
  )
}
