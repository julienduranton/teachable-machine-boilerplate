import * as speechCommands from "@tensorflow-models/speech-commands";

class soundClassifier {
  constructor(sampleLength, nclasses, modelName) {
    this.model = await this.loadTransferModel(modelName);
    this.NUM_FRAMES = Math.round(sampleLength / 23);
    this.INPUT_SHAPE = [this.NUM_FRAMES, 232, 1];
    this.NUM_CLASSES = nclasses;
    this.recognizer = speechCommands.create();
    this.examples = [];
  }
  async loadTransferModel(newModelName) {
    const baseRecognizer = speechCommands.create('BROWSER_FFT');
    await baseRecognizer.ensureModelLoaded();
    const transferRecognizer = baseRecognizer.createTransfer(newModelName);
    return transferRecognizer;    
  }

  save_model() {
  }

  addClass(className) {
    this.model.collectExample(className);
  }

  predict() {
    await this.model.listen(result => {
      // - result.scores contains the scores for the new vocabulary, which
      //   can be checked with:
      const words = transferRecognizer.wordLabels();
      // `result.scores` contains the scores for the new words, not the original
      // words.
      for (let i = 0; i < words; ++i) {
        console.log(`score for detection '${words[i]}' = ${result.scores[i]}`);
      }
    }, { probabilityThreshold: 0.75 });
    
  }
  loadWav(filename) {
    var req = new XMLHttpRequest();
    req.open("GET", filename, true);
    req.responseType = "arraybuffer";

    req.onload = function () {
      var arrayBuffer = req.response;
      if (arrayBuffer) {
        var byteArray = new Float32Array(arrayBuffer);
      }
    };
    req.send(null);

    // TODO spectrogram from arrayBuffer
    const mySpectrogramData = [];

    const x = tf.tensor4d(
      mySpectrogramData, [1].concat(this.model.modelInputShape().slice(1)));   
  }
  AudioSpectrogram()
  {
    var audioCtx = new AudioContext();
    var analyserNode = audioCtx.createAnalyser();
    const frqBuf = new Uint8Array(analyserNode.frequencyBinCount); // 1024
    const wfNumPts = 50*analyserNode.frequencyBinCount/128; // 400 +ve freq bins
    const wfBufAry = [frqBuf];
    analyserNode.getByteFrequencyData(frqBuf, 0);
  // const wf = new Waterfall(wfBufAry, wfNumPts, wfNumPts, "right", {});    
  // const canvas = document.getElementById(cvsID);
  // const ctx = canvas.getContext("2d");

  // this.playing = false;
  // this.begin = ()=>{
  //   wf.start();
  //   this.playing = true;
  //   this.drawOnScreen();
  // };

  // this.halt = ()=>{
  //   wf.stop();
  //   this.playing = false;
  // };

  // this.drawOnScreen = ()=>{
  //   analyserNode.getByteFrequencyData(frqBuf, 0);
  //   ctx.drawImage(wf.offScreenCvs, 0, 0);
  //   if (this.playing) requestAnimationFrame(this.drawOnScreen);
  // };
  }

  train() {
    await this.model.train({
      epochs: 25,
      callback: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
        }
      }
    });
  }
  
  // buildModel() {
  //   model = tf.sequential();
  //   model.add(
  //     tf.layers.depthwiseConv2d({
  //       depthMultiplier: 8,
  //       kernelSize: [this.NUM_FRAMES, 3],
  //       activation: "relu",
  //       inputShape: this.INPUT_SHAPE,
  //     })
  //   );
  //   model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  //   model.add(tf.layers.flatten());
  //   model.add(
  //     tf.layers.dense({ units: this.NUM_CLASSES, activation: "softmax" })
  //   );
  //   const optimizer = tf.train.adam(0.01);
  //   model.compile({
  //     optimizer,
  //     loss: "categoricalCrossentropy",
  //     metrics: ["accuracy"],
  //   });
  //   return this.model;
  // }

  // async train() {
  //   // toggleButtons(false);
  //   const ys = tf.oneHot(
  //     examples.map((e) => e.label),
  //     this.NUM_CLASSES
  //   );
  //   const xsShape = [examples.length, ...this.INPUT_SHAPE];
  //   const xs = tf.tensor(this.flatten(examples.map((e) => e.vals)), xsShape);

  //   await model.fit(xs, ys, {
  //     batchSize: 16,
  //     epochs: 10,
  //     callbacks: {
  //       onEpochEnd: (epoch, logs) => {
  //         document.querySelector("#console").textContent = `Accuracy: ${(
  //           logs.acc * 100
  //         ).toFixed(1)}% Epoch: ${epoch + 1}`;
  //       },
  //     },
  //   });
  //   tf.dispose([xs, ys]);
  //   // toggleButtons(true);
  // }
  // flatten(tensors) {
  //   const size = tensors[0].length;
  //   const result = new Float32Array(tensors.length * size);
  //   tensors.forEach((arr, i) => result.set(arr, i * size));
  //   return result;
  // }
  // toggleButtons(enable) {
  //   document.querySelectorAll("button").forEach((b) => (b.disabled = !enable));
  // }
  // collect(label) {
  //   if (this.recognizer.isListening()) {
  //     return this.recognizer.stopListening();
  //   }
  //   if (label == null) {
  //     return;
  //   }
  //   this.recognizer.listen(
  //     async ({ spectrogram: { frameSize, data } }) => {
  //       let vals = this.normalize(data.subarray(-frameSize * this.NUM_FRAMES));
  //       this.examples.push({ vals, label });
  //       document.querySelector(
  //         "#console"
  //       ).textContent = `${this.examples.length} examples collected`;
  //     },
  //     {
  //       overlapFactor: 0.999,
  //       includeSpectrogram: true,
  //       invokeCallbackOnNoiseAndUnknown: true,
  //     }
  //   );
  // }

  // predictSound() {
  //   if (this.recognizer.isListening()) {
  //     this.recognizer.stopListening();
  //     this.toggleButtons(true);
  //     document.getElementById("listen").textContent = "Listen";
  //     return;
  //   }
  //   this.toggleButtons(false);
  //   document.getElementById("listen").textContent = "Stop";
  //   document.getElementById("listen").disabled = false;

  //   this.recognizer.listen(
  //     async ({ spectrogram: { frameSize, data } }) => {
  //       const vals = normalize(data.subarray(-frameSize * this.NUM_FRAMES));
  //       const input = tf.tensor(vals, [1, ...this.INPUT_SHAPE]);
  //       const probs = this.model.predict(input);
  //       const predLabel = probs.argMax(1);
  //       await this.outputPrediction(predLabel);
  //       tf.dispose([input, probs, predLabel]);
  //     },
  //     {
  //       overlapFactor: 0.999,
  //       includeSpectrogram: true,
  //       invokeCallbackOnNoiseAndUnknown: true,
  //     }
  //   );
  // }
  // async outputPrediction(labelTensor) {
  //   const label = (await labelTensor.data())[0];
  //   document.getElementById("console").textContent = label;
  //   // if (label == 2) {
  //   //   return;
  //   // }
  //   let delta = 0.1;
  //   const prevValue = +document.getElementById("output").value;
  //   document.getElementById("output").value =
  //     prevValue + (label === 0 ? -delta : delta);
  // }

  // normalize(x) {
  //   const mean = -100;
  //   const std = 10;
  //   return x.map((x) => (x - mean) / std);
  // }
}
