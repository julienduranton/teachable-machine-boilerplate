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
      // const button = document.createElement("button");
      // button.innerText = "Train " + i;
      // div.appendChild(button);

      // Create training input
      const trainInput = document.createElement("input");
      trainInput.setAttribute("name", "Upload " + i);
      trainInput.setAttribute("id", "upload" + i);
      trainInput.setAttribute("type", "file");
      trainInput.setAttribute("multiple", "");
      div.appendChild(trainInput);

      // Listen for mouse events when clicking the button
      //button.addEventListener("mousedown", () => (this.training = i));
      //button.addEventListener("mouseup", () => (this.training = -1));

      trainInput.addEventListener("change", (e) => {
        this.training = i;
        console.log(e);
      });

      // Create info text
      const infoText = document.createElement("span");
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }

    // Create a container for the start button
    const startDiv = document.createElement("div");
    document.body.appendChild(startDiv);
    // Start prediction
    const trainButton = document.createElement("button");
    trainButton.innerText = "Train";
    // Create start button
    const startButton = document.createElement("button");
    startButton.innerText = "Start";

    startButton.addEventListener("mousedown", () => this.start());
    trainButton.addEventListener("mousedown", () => this.train());
    startDiv.appendChild(startButton);
    startDiv.appendChild(trainButton);
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();
    console.log(this.mobilenet);
  }

  start() {
    try {
      this.knn = loadClassifierFromLocalStorage();
      console.log(loadClassifierFromLocalStorage());
      console.log("Cached data loaded successfully");
    } catch (error) {
      console.log(error);
    }
    console.log(this.knn);
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.videoClip.play();
    this.videoPlaying = true;
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async train() {
    await this.loadImages();
    saveClassifierInLocalStorage(this.knn);
  }

  async loadImages() {
    console.log(this.mobilenet);
    for (let i = 0; i < NUM_CLASSES; i++) {
      var className = "upload" + i;
      const uploadedFiles = document.getElementById(className).files;
      for (let j = 0; j < uploadedFiles.length; j++) {

        let image = new Image();
        let uploadedImage = new Image();

        // 'conv_preds' is the logits activation of MobileNet.
        let logits;
        let infer;
        const mobilenet = this.mobilenet;
        const knn = this.knn;

        uploadedImage.onload = function() {
          image.src = this.src;
          infer = () => mobilenet.infer(image, "conv_preds")
          logits = infer();
          // Add current image to classifier
          knn.addExample(logits, i);
        }
        uploadedImage.src= URL.createObjectURL(uploadedFiles[j]);
      }
    }
  }

  async animate() {
    this.knn = loadClassifierFromLocalStorage();
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
      // this.knn.addExample(logits, );
      // }

      const numClasses = this.knn.getNumClasses();
      console.log(numClasses);
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


async function toDatasetObject(dataset) {
  const result = await Promise.all(
    Object.entries(dataset).map(async ([classId,value], index) => {
      const data = await value.data();
      return {
        classId: Number(classId),
        data: Array.from(data),
        shape: value.shape
      };
   })
  );
  return result;
};

function fromDatasetObject(datasetObject) {
  return Object.entries(datasetObject).reduce((result, [indexString, {data, shape}]) => {
    const tensor = tf.tensor2d(data, shape);
    const index = Number(indexString);
    result[index] = tensor;
    return result;
  }, {});
}

const storageKey = "knnClassifier";

async function saveClassifierInLocalStorage(classifier) {
  console.log('Inside');
  console.log(classifier);
  const dataset = classifier.getClassifierDataset();
  console.log(dataset);
  const datasetOjb = await toDatasetObject(dataset);
  console.log(datasetOjb);
  const jsonStr = await JSON.stringify(dataset);
  console.log("I love Json strings")
  console.log(jsonStr);
  //can be change to other source
  localStorage.setItem(storageKey, jsonStr);
}

// To be used later to load the knn
function loadClassifierFromLocalStorage() {
  const classifier = knnClassifier.create();
  const dataset = localStorage.getItem(storageKey);
  if (dataset) {
    console.log("I love Json")
    const datasetObj = JSON.parse(datasetJson);
    //const dataset = fromDatasetObject(datasetObj);
    classifier.setClassifierDataset(datasetObj);
  }
  return classifier;
}