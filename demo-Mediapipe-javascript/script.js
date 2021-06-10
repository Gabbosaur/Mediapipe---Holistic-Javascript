const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

var rep_counter = 0;
var flag = 0;

function find_angle(A, B, C) {
  var AB = Math.sqrt(Math.pow(B.x - A.x, 2) + Math.pow(B.y - A.y, 2));
  var BC = Math.sqrt(Math.pow(B.x - C.x, 2) + Math.pow(B.y - C.y, 2));
  var AC = Math.sqrt(Math.pow(C.x - A.x, 2) + Math.pow(C.y - A.y, 2));

  var radians = Math.acos((BC * BC + AB * AB - AC * AC) / (2 * BC * AB));
  var pi = Math.PI;
  return radians * (180 / pi);
}


function detectTPose(landmarks) {
  console.log(landmarks);
  if (typeof landmarks.poseLandmarks != "undefined") { // se non trova alcun scheletro
    // calcolo angolo braccia destra
    canvasCtx.font = "50px Arial";
    canvasCtx.fillStyle = "white";
    canvasCtx.strokeText("Body detected", 30, 80);
    var shoulder_right = landmarks.poseLandmarks[12];
    var elbow_right = landmarks.poseLandmarks[14];
    var wrist_right = landmarks.poseLandmarks[16];
    var angle_elbow_right = find_angle(shoulder_right, elbow_right, wrist_right);
    canvasCtx.font = "20px Arial";
    canvasCtx.fillText(Math.trunc(angle_elbow_right) + "°", canvasElement.width * elbow_right.x, canvasElement.height * elbow_right.y);

    // calcolo angolo braccia sinistra
    var angle_elbow_left = find_angle(landmarks.poseLandmarks[11], landmarks.poseLandmarks[13], landmarks.poseLandmarks[15]);
    canvasCtx.fillText(Math.trunc(angle_elbow_left) + "°", canvasElement.width * landmarks.poseLandmarks[13].x, canvasElement.height * landmarks.poseLandmarks[13].y);

    // spalla sinistra
    var angle_shoulder_left = find_angle(landmarks.poseLandmarks[23], landmarks.poseLandmarks[11], landmarks.poseLandmarks[13]);
    canvasCtx.fillText(Math.trunc(angle_shoulder_left) + "°", canvasElement.width * landmarks.poseLandmarks[11].x, canvasElement.height * landmarks.poseLandmarks[11].y);

    // spalla destra
    var angle_shoulder_right = find_angle(landmarks.poseLandmarks[24], landmarks.poseLandmarks[12], landmarks.poseLandmarks[14]);
    canvasCtx.fillText(Math.trunc(angle_shoulder_right) + "°", canvasElement.width * landmarks.poseLandmarks[12].x, canvasElement.height * landmarks.poseLandmarks[12].y);

    // gambe omesse
    /////////////////

    // stampa T pose
    var tollerance = 20;
    if (angle_elbow_left >= (180 - tollerance) && angle_elbow_left <= (180 + tollerance) && angle_elbow_right >= (180 - tollerance) && angle_elbow_right <= (180 + tollerance) && angle_shoulder_left <= (90 + tollerance) && angle_shoulder_left >= (90 - tollerance) && angle_shoulder_right <= (90 + tollerance) && angle_shoulder_right >= (90 - tollerance)) {
      canvasCtx.font = "50px Arial";
      canvasCtx.fillStyle = "blue";
      canvasCtx.textAlign = "center";
      canvasCtx.fillText("T-Pose found", canvasElement.width/2, 50);
      if (flag == 0) {
        rep_counter += 1;
        flag = 1;
      }
    } else {
      if (angle_shoulder_left < 20 && angle_shoulder_right < 20) {
        flag = 0;
      }
    }

    canvasCtx.font = "50px Arial";
    canvasCtx.fillStyle = "white";
    canvasCtx.textAlign = "left";
    canvasCtx.strokeText("Rep: " + rep_counter, 50, 150);

  } else {
    canvasCtx.font = "50px Arial";
    canvasCtx.fillStyle = "blue";
    canvasCtx.strokeText("No body detected", 50, 50);
  }


}

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image, 0, 0, canvasElement.width, canvasElement.height);
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
    { color: '#00FF00', lineWidth: 4 });
  drawLandmarks(canvasCtx, results.poseLandmarks,
    { color: '#FF0000', lineWidth: 2 });
  detectTPose(results);
  canvasCtx.restore();
}

const pose = new Pose({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
  }
});
pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
pose.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({ image: videoElement });
  },
  width: 1280,
  height: 720
});
camera.start();