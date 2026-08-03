# Modifications

The source is `Supertone/supertonic-3` revision
`724fb5abbf5502583fb520898d45929e62f02c0b`.

- duration_predictor.onnx: unchanged FP32
- text_encoder.onnx: unchanged FP32
- vector_estimator.onnx: modified by lowering eligible constant-weight affine layers to `MatMul` and encoding 95 weights as asymmetric Q4 `com.microsoft::MatMulNBits` with block size 16. The final projection and depthwise convolutions remain FP32.
- vocoder.onnx: modified by lowering eligible affine layers and encoding 18 weights as asymmetric Q4 `com.microsoft::MatMulNBits` with block size 16.

The vocoder keeps these audited boundary layers in FP32:

- `/decoder/embed/net/Conv`
- `/decoder/convnext.9/pwconv1/Conv`
- `/decoder/convnext.9/pwconv2/Conv`
- `/decoder/head/layer1/net/Conv`
- `/decoder/head/layer2/Conv`

The modified ONNX files contain `parapper.*` metadata with the source revision,
component name, quantization contract, and derivative notice.
