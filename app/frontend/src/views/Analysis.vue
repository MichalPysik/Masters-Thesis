<script setup>
import { ref, computed, onMounted } from "vue";
import { useRoute } from 'vue-router';
const route = useRoute();

// Start analysis variables (editable)
const video_name = ref("");
const start_seconds = ref(0);
const end_seconds = ref(10);
const sample_frames = ref(16);

// Current analysis variables (constants, copied from start variables)
const locked_video_name = ref("");
const locked_start_seconds = ref(0);
const locked_end_seconds = ref(10);
const locked_sample_frames = ref(16);

// List of video names that are in the bucket
const video_names = ref([]);

// Computed list of video name suggestions matching current prefix (case insensitive)
const suggestions = computed(() => {
    if (!video_name.value) return video_names.value;
    const prefix = video_name.value.toLowerCase();
    return video_names.value.filter(name => name.toLowerCase().startsWith(prefix));
});

// Helper function to convert seconds to HH:MM:SS.mmm format
function seconds_to_timestamp(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00:00.000";
    const integerPart = Math.floor(seconds);
    const hours = Math.floor(integerPart / 3600);
    const minutes = Math.floor((integerPart % 3600) / 60);
    const secs = integerPart % 60;
    const ms = Math.floor((seconds - integerPart) * 1000);
    const pad2 = num => String(num).padStart(2, "0");
    const pad3 = num => String(num).padStart(3, "0");
    return `${pad2(hours)}:${pad2(minutes)}:${pad2(secs)}.${pad3(ms)}`;
}

// Computed properties for the formatted timestamps and sampling FPS
const start_timestamp = computed(() => seconds_to_timestamp(Number(start_seconds.value)));
const end_timestamp = computed(() => seconds_to_timestamp(Number(end_seconds.value)));
const sample_fps = computed(() => {
    const duration = end_seconds.value - start_seconds.value;
    return (sample_frames.value / duration).toFixed(2);
});

// Computed properties for locked values
const locked_start_timestamp = computed(() => seconds_to_timestamp(Number(locked_start_seconds.value)));
const locked_end_timestamp = computed(() => seconds_to_timestamp(Number(locked_end_seconds.value)));
const locked_sample_fps = computed(() => {
    const duration = locked_end_seconds.value - locked_start_seconds.value;
    return (locked_sample_frames.value / duration).toFixed(2);
});

// Video URL and video player reference variables
const video_url = ref("");
const video_player = ref(null);

// When the video metadata is loaded, set the current time to start_seconds
function on_loaded_metadata() {
    if (video_player.value) {
        video_player.value.currentTime = Number(start_seconds.value);
    }
}

// Chat conversation state and current chat input
const conversation = ref([]);
const chat_input = ref("");
const is_loading = ref(false);

// Helper function to extract the text from a message's content array
function extract_message_text(message) {
    if (!message.content || !Array.isArray(message.content)) return "";
    let fullText = message.content
        .filter(item => item.type === "text")
        .map(item => item.text)
        .join(" ");

    // Remove instructions from the user's message
    if (message.role === "user") {
        const marker = "This is the user's query:";
        const markerIndex = fullText.indexOf(marker);
        if (markerIndex !== -1) {
            fullText = fullText.substring(markerIndex + marker.length).trim();
        }
    }
    return fullText;
}

// Function to send chat messages
async function send_chat_message() {
    const query = chat_input.value.trim();
    if (!query) return;

    let endpoint = "";
    let payload = {};
    is_loading.value = true;

    // If conversation is empty, it's the first message
    if (conversation.value.length === 0) {
        endpoint = `/api/start-video-analysis/${locked_video_name.value}`;
        payload = {
            query: query,
            start_timestamp: locked_start_seconds.value,
            end_timestamp: locked_end_seconds.value,
            num_frames: locked_sample_frames.value
        };
    } else {
        endpoint = `/api/continue-video-analysis`;
        payload = {
            query: query,
            existing_conversation: conversation.value
        };
    }

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        // Update conversation with the returned conversation (keeping the full history)
        conversation.value = data.conversation;
    } catch (error) {
        console.error("Error sending chat message:", error);
        alert("Failed to send chat message.");
    }

    // Clear input after sending
    chat_input.value = "";
    is_loading.value = false;
}

// Function to setup new analysis on button click
async function setup_new_analysis() {
    const confirmed = confirm("Are you sure you want to setup a new analysis? If an existing conversation exists, it will be lost.");
    if (!confirmed) return;

    if (!video_names.value.includes(video_name.value)) {
        alert("Invalid video name.");
        return;
    }

    if (start_seconds.value >= end_seconds.value) {
        alert("End timestamp must be greater than start timestamp.");
        return;
    }

    // Freeze the current values into locked variables
    locked_video_name.value = video_name.value;
    locked_start_seconds.value = start_seconds.value;
    locked_end_seconds.value = end_seconds.value;
    locked_sample_frames.value = sample_frames.value;

    // Clear conversation
    conversation.value = [];
    chat_input.value = "";

    // Fetch the video URL
    try {
        const response = await fetch(`/api/video-url/${video_name.value}`);
        const data = await response.json();
        video_url.value = data.url;
    } catch (error) {
        console.error("Error fetching video URL:", error);
    }
}

// On mount, handle possible redirect from search, and fetch the list of bucket videos 
onMounted(async () => {
    // Check if redirected from search with video name and timestamps
    if (route.query.video_name) {
        video_name.value = route.query.video_name;
        start_seconds.value = Number(route.query.start_seconds) || start_seconds.value;
        end_seconds.value = Number(route.query.end_seconds) || end_seconds.value;
    }

    // Fetch the list of bucket videos
    try {
        const response = await fetch('/api/data/list-all');
        const data = await response.json();
        // Extract video_name where in_bucket is true
        video_names.value = data
            .filter(entry => entry.in_bucket)
            .map(entry => entry.video_name)
            .sort((a, b) => a.localeCompare(b)); // Sort alphabetically
    } catch (error) {
        console.error("Error fetching video list:", error);
    }
});
</script>


<template>
    <div class="view-container">
        <!-- Parameters for starting new conversation -->
        <div class="parameters-bar">
            <div class="parameters-bar-group">
                <label for="video-name-input">Video:</label>
                <input type="text" id="video-name-input" placeholder="Enter video name..." v-model="video_name"
                    list="video-name-list" />
                <!-- The datalist provides suggestions of existing bucket videos -->
                <datalist id="video-name-list">
                    <option v-for="name in suggestions" :key="name" :value="name" />
                </datalist>
            </div>
            <div class="parameters-bar-group">
                <label for="start-timestamp-input">From:</label>
                <input id="start-timestamp-input" class="quad-digit-input" type="number" step="1" min="0" v-model="start_seconds" />
                <p>seconds ({{ start_timestamp }})</p>
            </div>
            <div class="parameters-bar-group">
                <label for="end-timestamp-input">To:</label>
                <input id="end-timestamp-input" class="quad-digit-input" type="number" step="1" min="1" v-model="end_seconds" />
                <p>seconds ({{ end_timestamp }})</p>
            </div>
            <div class="parameters-bar-group">
                <label for="sample-frames-input">Frames:</label>
                <input type="number" class="double-digit-input" id="sample-frames-input" min="1" max="32" step="1" v-model.number="sample_frames" />
                <p>({{ sample_fps }} FPS)</p>
            </div>
            <button @click="setup_new_analysis">New analysis</button>
        </div>

        <!-- Active analysis: video player, frozen parameters, and chat -->
        <div v-if="video_url" class="active-analysis-container">
            <video ref="video_player" :src="video_url" controls @loadedmetadata="on_loaded_metadata"></video>

            <div class="active-analysis-parameters">
                <p><strong>Video:</strong> {{ locked_video_name }}</p>
                <p><strong>Segment:</strong> {{ locked_start_timestamp }} to {{ locked_end_timestamp }}</p>
                <p><strong>Sampled frames:</strong> {{ locked_sample_frames }} ({{ locked_sample_fps }} FPS)</p>
            </div>

            <div class="chat-container">
                <div class="messages">
                    <div v-for="(msg, index) in conversation" :key="index" :class="['chat-message', msg.role]">
                        <p>{{ extract_message_text(msg) }}</p>
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" v-model="chat_input" @keyup.enter="send_chat_message"
                        placeholder="Type your message..." />
                    <button @click="send_chat_message" :disabled="is_loading">Send</button>
                </div>
                <div class="chat-loading-indicator" v-if="is_loading">Loading assistant response...</div>
            </div>
        </div>
    </div>
</template>
