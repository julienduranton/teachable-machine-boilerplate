// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as knnClassifier from "@tensorflow-models/knn-classifier";
import * as mobilenetModule from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create a video element to display video.avi
    this.videoClip = document.createElement("video");
    this.videoClip.setAttribute("controls", "");
    this.videoClip.setAttribute("width", "256px");
    this.videoClip.setAttribute("height", "144px");
    this.videoSource = document.createElement("source");
    this.videoSource.setAttribute("src", "/assets/video.mp4");
    this.videoSource.setAttribute("type", "video/mp4");
    this.videoClip.playbackRate = 2;

    // Create video element that will contain the webcam image
    this.video = document.createElement("video");
    this.video.setAttribute("autoplay", "");
    this.video.setAttribute("playsinline", "");

    // Add video element to DOM
    document.body.appendChild(this.videoClip);
    this.videoClip.appendChild(this.videoSource);
    document.body.appendChild(this.video);

    // Create training buttons and info texts
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement("div");
      document.body.appendChild(div);
      div.style.marginBottom = "10px";

      // Create training button
      const button = document.createElement("button");
      button.innerText = "Train " + i;
      div.appendChild(button);

      // Create training input

      const trainInput = document.createElement("input");
      trainInput.setAttribute("name", "Upload " + i);
      trainInput.setAttribute("id", "upload" + i);
      trainInput.setAttribute("type", "file");
      div.appendChild(trainInput);

      // Listen for mouse events when clicking the button
      button.addEventListener("mousedown", () => (this.training = i));
      button.addEventListener("mouseup", () => (this.training = -1));

      trainInput.addEventListener("change", () => (this.training = i));

      // Create info text
      const infoText = document.createElement("span");
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }

    // Create a container for the start button
    const startDiv = document.createElement("div");
    document.body.appendChild(startDiv);
    // Create start button
    const startButton = document.createElement("button");
    startButton.innerText = "Start";

    startButton.addEventListener("mousedown", () => this.start());
    startDiv.appendChild(startButton);

    // Setup webcam
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener(
          "playing",
          () => (this.videoPlaying = true)
        );
        this.video.addEventListener(
          "paused",
          () => (this.videoPlaying = false)
        );
      });
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    // this.start();
  }

  async start() {
    if (this.timer) {
      this.stop();
    }
    await this.loadImages();
    this.video.play();
    this.videoClip.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async loadImages() {
    for (let i = 0; i < NUM_CLASSES; i++) {
      var className = "upload" + i;
      const uploadedImages = document.getElementById(className).files;
      for (let j = 0; j < uploadedImages.length; j++) {
        const image = tf.fromPixels(uploadedImages[j]);
        let logits;
        // 'conv_preds' is the logits activation of MobileNet.
        const infer = () => this.mobilenet.infer(image, "conv_preds");
        logits = infer();
        // Add current image to classifier
        this.knn.addExample(logits, i);
      }
    }
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.videoClip);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, "conv_preds");

      // Train class if one of the buttons is held down
      //if (this.training != -1) {
      // logits = infer();

      // Add current image to classifier
      //  this.knn.addExample(logits, this.training);
      //}

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {
        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {
          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = "bold";
          } else {
            this.infoTexts[i].style.fontWeight = "normal";
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${
              res.confidences[i] * 100
            }%`;
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener("load", () => new Main());
