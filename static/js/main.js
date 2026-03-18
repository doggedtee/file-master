/**
 * main.js
 * UI logic: file upload, message display, requests to Flask.
 */

// DOM elements

const uploadScreen    = document.getElementById("uploadScreen");
const processingScreen= document.getElementById("processingScreen");
const chatScreen      = document.getElementById("chatScreen");
const inputArea       = document.getElementById("inputArea");
const dropZone        = document.getElementById("dropZone");
const fileInput       = document.getElementById("fileInput");
const messages        = document.getElementById("messages");
const userInput       = document.getElementById("userInput");
const btnAsk          = document.getElementById("btnAsk");
const btnNew          = document.getElementById("btnNew");
const subtitle        = document.getElementById("subtitle");
const processingStep  = document.getElementById("processingStep");
const progressFill    = document.getElementById("progressFill");


// State

let isAsking = false;


// Screen switching

function showScreen(name) {
  uploadScreen.style.display     = name === "upload"     ? "flex" : "none";
  processingScreen.style.display = name === "processing" ? "flex" : "none";
  chatScreen.style.display       = name === "chat"       ? "flex" : "none";
  inputArea.style.display        = name === "chat"       ? "block": "none";
  btnNew.classList.toggle("visible", name === "chat");
}


// Upload & processing

async function handleFile(file) {
  if (!file || !file.name.endsWith(".pdf")) {
    alert("Please upload a PDF file.");
    return;
  }

  showScreen("processing");
  setProgress("Uploading PDF to server...", 10);

  const formData = new FormData();
  formData.append("file", file);

  try {
    setProgress("Extracting text from PDF...", 25);

    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Upload failed");

    setProgress(`Building embeddings for ${data.chunk_count} chunks...`, 60);

    // Small delay so the user can see the progress
    await sleep(800);
    setProgress("Ready!", 100);
    await sleep(400);

    // Switch to chat screen
    subtitle.textContent = `"${data.filename}" · ${data.chunk_count} chunks indexed`;
    showScreen("chat");
    addMessage("assistant", `Document loaded! I've indexed **${data.chunk_count} chunks** from "${data.filename}". Ask me anything about it.`);

  } catch (err) {
    showScreen("upload");
    alert("Error: " + err.message);
  }
}

function setProgress(step, pct) {
  processingStep.textContent   = step;
  progressFill.style.width     = pct + "%";
}


// Chat

function addMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = "message";

  const avatar = document.createElement("div");
  avatar.className = `message__avatar message__avatar--${role}`;
  avatar.textContent = role === "user" ? "YOU" : "AI";

  const text = document.createElement("div");
  text.className = `message__content message__content--${role}`;
  text.textContent = content.replace(/\*\*(.*?)\*\*/g, "$1");

  msg.appendChild(avatar);
  msg.appendChild(text);
  messages.appendChild(msg);

  // Scroll to bottom
  msg.scrollIntoView({ behavior: "smooth", block: "end" });

  return msg;
}

function addThinkingIndicator() {
  const msg = document.createElement("div");
  msg.className = "message";
  msg.id = "thinking";

  const avatar = document.createElement("div");
  avatar.className = "message__avatar message__avatar--assistant";
  avatar.textContent = "AI";

  const text = document.createElement("div");
  text.className = "message__content message__content--thinking";
  text.textContent = "Searching document...";

  msg.appendChild(avatar);
  msg.appendChild(text);
  messages.appendChild(msg);
  msg.scrollIntoView({ behavior: "smooth", block: "end" });
}

function removeThinkingIndicator() {
  const el = document.getElementById("thinking");
  if (el) el.remove();
}

async function sendMessage() {
  const question = userInput.value.trim();
  if (!question || isAsking) return;

  isAsking = true;
  btnAsk.disabled = true;
  userInput.value = "";
  userInput.style.height = "auto";

  addMessage("user", question);
  addThinkingIndicator();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    removeThinkingIndicator();

    if (!res.ok) throw new Error(data.error || "Request failed");

    addMessage("assistant", data.answer);

  } catch (err) {
    removeThinkingIndicator();
    addMessage("assistant", "⚠️ Error: " + err.message);
  } finally {
    isAsking = false;
    btnAsk.disabled = false;
  }
}


// Event listeners

// Drag & drop
dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop",      (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  handleFile(e.dataTransfer.files[0]);
});

// Click to browse
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

// Send message
btnAsk.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
});

// New document
btnNew.addEventListener("click", () => {
  messages.innerHTML = "";
  subtitle.textContent = "Upload a PDF to begin";
  showScreen("upload");
});


// Utils

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}