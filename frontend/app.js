//Frontend JavaScript for Brain Tumor Classification App
const API = ""; 

async function checkHealth() {
  const statusElem = document.getElementById("modelStatus");
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    statusElem.textContent = data.status;
    statusElem.className = data.status.toLowerCase().includes("healthy") || data.status.toLowerCase().includes("online")
      ? "px-3 py-1 rounded-full text-sm font-semibold bg-emerald-500 text-slate-900" 
      : "px-3 py-1 rounded-full text-sm font-semibold bg-rose-500 text-white";
  } catch (err) {
    statusElem.textContent = "Offline (API Unreachable)";
    statusElem.className = "px-3 py-1 rounded-full text-sm font-semibold bg-rose-500 text-white";
  }
}

async function uploadImage() {
  const fileInput = document.getElementById("imageInput");
  const resultBox = document.getElementById("resultBox");
  const predText = document.getElementById("predictionText");
  const confText = document.getElementById("confidenceText");

  if (!fileInput.files[0]) {
    alert("Please select an MRI image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    predText.textContent = "Processing MRI...";
    confText.textContent = "";
    resultBox.classList.remove("hidden");

    const res = await fetch(`${API}/predict`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Prediction request failed");

    const data = await res.json();
    predText.textContent = data.prediction.toUpperCase();
    confText.textContent = `Confidence: ${data.confidence}%`;
  } catch (err) {
    predText.textContent = "Error processing image";
    confText.textContent = err.message;
  }
}

async function triggerRetrain() {
  const status = document.getElementById("retrainStatus");
  status.textContent = "Initiating retraining trigger...";
  try {
    const res = await fetch(`${API}/retrain`, { method: "POST" });
    const data = await res.json();
    status.textContent = data.message || "Retraining triggered.";
  } catch (err) {
    status.textContent = "Failed to trigger retraining.";
  }
}

// Initial check
checkHealth();