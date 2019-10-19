
const knnClassifier = ml5.KNNClassifier();
let featureExtractor;
let counts;
let k = 15;
let exLim = 150;
let currentRes = "";
let video;
let cnt = 0;
let conf = 0;
let target = ["smart", "simple", "strong", "stylish", "idk"];
//
function setup() {
  createCanvas(1280, 720);
  featureExtractor = ml5.featureExtractor('MobileNet', () => {
    console.log("model loaded")
  });
  video = createCapture(VIDEO, () => {
    console.log("video loaded")
  });
  
}

function draw() {
  background(220);
  counts = knnClassifier.getCountByLabel();
  image(video, 0, 0, video.width, video.height);
  video.hide();
  noFill();
  rect(0, 500, 160, 30);
  fill(255, 0, 0);
  rect(0, 500, 160*(cnt/(exLim*5)), 30);
  textSize(32);
  if(currentRes != "") text("current result is: " + currentRes + " with " + (conf[currentRes]*100) + "%", 150, 530);
  //console.log(video.size());
  let key1 = 49;
  for(let i = 0 ; i < 5 ; i ++){
    if(keyIsDown(key1 + i)){
      addEx(target[i]);
    }
  }
  //if(mouseIsPressed) addEx(target[4]);
  if(keyIsDown(32)) predict();
  //if(keyIsDown(72)) saveMyKNN();
}

function keyPressed(){
  if(keyCode == 72)  saveMyKNN();
}
function addEx(label){
  if((counts[label] || 0) < exLim){
    if(counts[label] == exLim - 1) console.log(label + " exceeded");
    console.log("adding " + label);
    cnt ++;
    const features = featureExtractor.infer(video);
    knnClassifier.addExample(features, label);
  }
}

function predict(label){
  const features = featureExtractor.infer(video);
  knnClassifier.classify(features, k, gotResults);
}

function gotResults(err, result){
  if(err) console.log(err);
  if (result.confidencesByLabel){
    conf = result.confidencesByLabel;
    if(result.label){
      currentRes = result.label;
      console.log(result.label);
    }
  }
  predict();
}

function saveMyKNN() {
  knnClassifier.save('myKNNDataset');
}

function loadMyKNN() {
  knnClassifier.load('./myKNNDataset.json', updateCounts);
}
