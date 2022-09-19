/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2022 @wyvern. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 * 
 * MODIFICAIONS MADE TO ORIGINAL:
 * Removed tfvis and using chart.js instead
 * Remove teh option for convnet models
 * restyled all things and removed mdc web
 * added download option for training progress
 * 
 */

import {IMAGE_H, IMAGE_SIZE, IMAGE_W, MnistData} from './data.js';
import "./ui.js";
let data = new MnistData();

const statusElement = document.getElementById('status');
const downloadButton = document.getElementById("download");
const trainButton = document.getElementById('train');
const layerCountInput = document.querySelector("input#hiddenlayers");
layerCountInput.value = 1;

Chart.defaults.color = "#fff";
Chart.defaults.borderColor = "#555";
const chart = new Chart(document.querySelector("#plot").getContext("2d"), {
    type: "scatter",
    data:{
        datasets:[{
            label: "Test",
            data: [],
            backgroundColor: "#7a55eb",
            borderColor: "#7a55eb"
        },{
            label: "Validation",
            data: []
        }]
    },
    options:{
        responsive: true,
        x: {
            type: 'linear',
            position: 'bottom'
        },
        backgroundColor: "#ffffff",
        color: "#ffffff",
        borderColor: "#ffffff",
        scales: {
            x: {
                beginAtZero: true,
                title:{
                    align: "center",
                    text: "Batches Elapsed (unitless)",
                    color: "#fff",
                    display: true
                }
            },
            y: {
                beginAtZero: true,
                title:{
                    align: "center",
                    text: "Accuracy (unitless)",
                    color: "#fff",
                    display: true
                }
            }
        }
    }
});

function logStatus(text){
    statusElement.innerText = text;
}
let model;
function createModel(hiddenLayers = 1) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
    let ratio = ( 10 / IMAGE_SIZE ) ** (1 / (1 + hiddenLayers));
    for(let i = 1; i <= hiddenLayers; i++){
        model.add(tf.layers.dense({
            units: Math.round(IMAGE_SIZE * ratio ** i),
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2(),
            biasRegularizer: tf.regularizers.l2()
        }));
    }
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return model;
}

async function train(model) {
    let csvLog = "Batch #, Accuracy, Loss\n";
    logStatus(`Training model on ${tf.getBackend()} as backend ...`);
    
    let batch = 0;
    
    const batchSize = 400;
    const validationSplit = 0.1;
    const trainEpochs = 3;
    
    const trainData = data.getTrainData();
    const testData = data.getTestData();
    const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;
    
    let last_val_acc;
    
    await model.fit(trainData.xs, trainData.labels, {
        batchSize,
        validationSplit,
        epochs: trainEpochs,
        callbacks: {
            onBatchEnd: async (_, {acc, loss}) => {
                logStatus(`Training... (${(batch / totalNumBatches * 100).toFixed(1)}% complete).`);
                csvLog += `${batch}, ${acc}, ${loss}\n`;
                chart.data.datasets[0].data.push({x: batch, y:acc});
                if(batch % 10 === 0){
                    chart.update('none');
                }
                batch += 1;
                //await tf.nextFrame();
            },
            onEpochEnd: async function(epoch, {acc, loss, val_loss, val_acc}){
                chart.data.datasets[1].data.push({x: batch, y:val_acc});
                chart.update("none");
                last_val_acc = val_acc;
            }
        }
    });
    const testResult = model.evaluate(testData.xs, testData.labels);
    const testAccPercent = testResult[1].dataSync()[0] * 100;
    logStatus(`Final test acc.: ${testAccPercent.toFixed(1)}%, Final val. acc.: ${(last_val_acc * 100).toFixed(1)}`);
    return csvLog;
}

trainButton.addEventListener("click", async () => {
    //reset chart
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update('none');
    
    //reset status log
    statusElement.innerText = "";
    
    //reset download links
    URL.revokeObjectURL(downloadButton.getAttribute("x-href") || "");
    downloadButton.removeAttribute("disabled");
    downloadButton.removeAttribute("x-href");
    downloadButton.removeAttribute("x-download");
    
    //start training
    trainButton.setAttribute('disabled', true);
    downloadButton.setAttribute('disabled', true);
    await tf.setBackend('webgl');
    
    logStatus('Creating model...');
    let hiddenLayers = parseInt(layerCountInput.value);
    hiddenLayers = hiddenLayers === NaN ? 1 : hiddenLayers;
    model = createModel(hiddenLayers);
    model.summary();
    
    logStatus('Loading MNIST data...');
    await data.load();
    
    logStatus('Starting model training...');
    let csvLog = await train(model);
    let blob = new Blob([csvLog], {type: "text/csv"})
    let url = URL.createObjectURL(blob);
    
    downloadButton.removeAttribute("disabled");
    downloadButton.setAttribute("x-href", url);
    downloadButton.setAttribute("x-download", `L${hiddenLayers}T${Date.now() % 1e4}.csv`);
    trainButton.removeAttribute("disabled");
    
    csvLog = null;
    blob = null;
    url = null;
});

downloadButton.addEventListener("click", async function () {
    let a = document.createElement('a');
    a.href = this.getAttribute("x-href");
    a.download = this.getAttribute("x-download");
    a.click();
    a = null;
})
export function predict(array){
    let input = tf.reshape(tf.tensor1d(array), [1, 28, 28, 1]);
    if(!model){
        return false;
    }
    return model.predict(input, {
        batchSize: 1
    }).arraySync()[0];
}