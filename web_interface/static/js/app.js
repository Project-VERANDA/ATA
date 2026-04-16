document.addEventListener('DOMContentLoaded', () => {
    loadTranscripts();
    startStatusPolling();
});

async function uploadVideo() {
    const form = document.getElementById('uploadForm');
    const input = document.getElementById('videoInput');
    const statusDiv = document.getElementById('uploadStatus');
    const btn = document.getElementById('uploadBtn');

    if (!input.files[0]) {
        statusDiv.innerText = "Please select a file.";
        statusDiv.style.color = "red";
        return;
    }

    const formData = new FormData(form);
    btn.disabled = true;
    statusDiv.innerText = "Uploading...";
    statusDiv.style.color = "#333";

    try {
        const response = await fetch('/upload_video', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.success) {
            statusDiv.innerText = `Success: ${data.filename}`;
            statusDiv.style.color = "green";
            input.value = ''; // Reset input
        } else {
            statusDiv.innerText = `Error: ${data.error}`;
            statusDiv.style.color = "red";
        }
    } catch (err) {
        statusDiv.innerText = "Network error.";
        statusDiv.style.color = "red";
    } finally {
        btn.disabled = false;
    }
}

// Attach submit event
document.getElementById('uploadForm').addEventListener('submit', (e) => {
    e.preventDefault();
    uploadVideo();
});

async function runPipeline() {
    const btn = document.getElementById('runBtn');
    if (btn.disabled) return;

    btn.disabled = true;
    btn.innerText = "Starting...";

    try {
        const response = await fetch('/run_pipeline', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            alert("Pipeline started!");
        } else {
            alert(`Error: ${data.error}`);
            btn.disabled = false;
            btn.innerText = "Run Pipeline";
        }
    } catch (err) {
        alert("Network error starting pipeline.");
        btn.disabled = false;
        btn.innerText = "Run Pipeline";
    }
}

async function loadTranscripts() {
    try {
        const response = await fetch('/get_transcripts');
        const data = await response.json();
        const listDiv = document.getElementById('transcriptList');

        if (data.transcripts.length === 0) {
            listDiv.innerHTML = '<p>No transcripts found yet.</p>';
            return;
        }

        listDiv.innerHTML = data.transcripts.map(t => `
            <div class="list-item">
                <span>${t.filename} (${Math.round(t.size/1024)} KB)</span>
                <a href="/download_transcript/${t.filename}" target="_blank">Download</a>
            </div>
        `).join('');
    } catch (err) {
        console.error("Failed to load transcripts", err);
    }
}

function startStatusPolling() {
    setInterval(async () => {
        try {
            const response = await fetch('/pipeline_status');
            const data = await response.json();
            
            const statusText = document.getElementById('statusText');
            const progressBar = document.getElementById('progressBar');
            const errorLog = document.getElementById('errorLog');

            statusText.innerText = data.current_step || 'Idle';
            progressBar.style.width = `${data.progress}%`;

            if (data.errors && data.errors.length > 0) {
                errorLog.style.display = 'block';
                errorLog.innerText = data.errors.join('\n');
            } else {
                errorLog.style.display = 'none';
            }

            // Re-enable button if not running
            const btn = document.getElementById('runBtn');
            if (!data.running && btn.disabled) {
                btn.disabled = false;
                btn.innerText = "Run Pipeline";
            }
        } catch (err) {
            console.error("Status poll failed", err);
        }
    }, 2000);
}
