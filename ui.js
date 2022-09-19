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
 */
import {predict} from "./index.js";
let monitors = document.querySelector(".monitors");

monitors.addEventListener("click", e => {
    console.log("trig")
    let ele = e.target;
    let href = ele.getAttribute && ele.getAttribute("x-href");
    if(!href) return;
    document.querySelector(".monitors a.active").classList.remove("active");
    document.querySelector(".monitor.active").classList.remove("active");
    document.getElementById(href).classList.add("active");
    ele.classList.add("active");
})

//handle canvas drawing stuff
const doodleCanvas = document.querySelector("canvas.doodle");
const ctx = doodleCanvas.getContext("2d");
ctx.lineWidth = 3;
let mode = null;
let lastX = null;
let lastY = null;
doodleCanvas.addEventListener("mousedown", e => {
    if(e.button === 2){
        mode = "erasing";
    }else if(e.button === 0){
        mode = "painting";
    }
});
doodleCanvas.addEventListener("mouseup", () => mode = null);
doodleCanvas.addEventListener("mousemove", ({offsetX, offsetY, ctrlKey}) => {
    let px = offsetX * 28 / doodleCanvas.offsetWidth;
    let py = offsetY * 28 / doodleCanvas.offsetHeight;
    if(lastX === null || lastY === null){

    }else if(mode != null){
        if(mode  === "painting" && (!ctrlKey)){
            ctx.strokeStyle = "#000";
        }else{
            ctx.strokeStyle = "#fff"
        }
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(px, py);
        ctx.stroke();
    }
    lastX = px;
    lastY = py;
})

const predictButton = document.getElementById("predict");
predictButton.addEventListener("click", () => {
    let res = []
    let {data} = ctx.getImageData(0, 0, 28, 28);
    for(let i = 0; i < data.length; i += 4){
        if(data[i] === 0 && data[i + 3] !== 0){
            res.push(1);
        }else{
            res.push(0);
        }
    }
    let probs = predict(res);
    if(probs == false){
        alert("Please train the network first, and then try again");
        return;
    }
    console.log(probs);
    for(let i = 0; i < 10; i++){
        document.querySelector(`#probabilities tr:nth-child(${i + 2}) td:nth-child(2)`).innerText = (probs[i] * 100).toFixed(0);
    }
})

const clearButton = document.getElementById("cleardoodle");
clearButton.addEventListener("click", () => {
    ctx.clearRect(0, 0, 28, 28);
})