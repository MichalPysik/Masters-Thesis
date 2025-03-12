<script setup>
import { ref, computed, onMounted } from "vue";

const max_videos_per_page = 25;

// Variables for video management
const selected_file = ref(null);
const sampling_fps = ref(1.0);
const bucket_only = ref(false);
const videos = ref([]);
const current_page = ref(1);
const force_bucket_mirror = ref(false);

// Pagination (only show a limited number of entries per page)
const paginated_videos = computed(() => {
    const start = (current_page.value - 1) * max_videos_per_page;
    return videos.value.slice(start, start + max_videos_per_page);
});

const total_pages = computed(() =>
    Math.ceil(videos.value.length / max_videos_per_page)
);

// File selector for video upload
function handle_file_change(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
        selected_file.value = files[0];
    }
}

// Upload video to the server
async function upload_video() {
    if (!selected_file.value || !sampling_fps.value) {
        alert("Please select a video file and enter Sampling FPS.");
        return;
    }

    const form_data = new FormData();
    form_data.append("video_file", selected_file.value);
    form_data.append("sampling_fps", sampling_fps.value);
    form_data.append("bucket_only", bucket_only.value);

    try {
        const res = await fetch("/api/data/upload-video", {
            method: "POST",
            body: form_data,
        });

        if (!res.ok) throw new Error("Upload failed.");

        alert("Upload successful.");
        await fetch_videos(); // Refresh video list after successful upload
    } catch (err) {
        console.error(err);
        alert("Upload failed.");
    }
}

// Fetch/refresh the list of videos in the Minio bucket and Milvus collection
async function fetch_videos() {
    try {
        const res = await fetch("/api/data/list-all");
        if (!res.ok) throw new Error("Failed to fetch videos.");
        videos.value = await res.json();

        // Sort videos by name alphabetically
        videos.value = videos.value.sort((a, b) =>
            a.video_name.localeCompare(b.video_name)
        );
    } catch (err) {
        console.error(err);
        alert("Failed to fetch videos.");
    }
}

// Delete a video button handler
async function delete_video(video_name) {
    try {
        const res = await fetch(`/api/data/delete-video/${video_name}`, {
            method: "DELETE",
        });

        if (!res.ok) throw new Error("Failed to delete video.");

        alert(`Video ${video_name} deleted successfully.`);
        await fetch_videos(); // Refresh video list after deletion
    } catch (err) {
        console.error(err);
        alert("Failed to delete video.");
    }
}

// Synchronize a video button handler
async function synchronize_video(video_name) {
    try {
        const res = await fetch(`/api/data/synchronize-video/${video_name}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                sampling_fps: sampling_fps.value,
            }),
        });

        if (!res.ok) throw new Error("Failed to synchronize video.");

        alert(`Video ${video_name} synchronized successfully.`);
        await fetch_videos(); // Refresh video list after synchronization
    } catch (err) {
        console.error(err);
        alert("Failed to synchronize video.");
    }
}

// Synchronize all data button handler
async function synchronize_all_data() {
    const confirmed = confirm("Are you sure you want to synchronize all data?");
    if (!confirmed) return;

    try {
        const res = await fetch("/api/data/synchronize-all", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                force_bucket_mirror: force_bucket_mirror.value,
                sampling_fps: sampling_fps.value,
            }),
        });

        if (!res.ok) throw new Error("Failed to synchronize all data.");

        alert("All data synchronized successfully.");
        await fetch_videos(); // Refresh video list after synchronization
    } catch (err) {
        console.error(err);
        alert("Failed to synchronize all data.");
    }
}

// Delete all data button handler
async function delete_all_data() {
    const confirmed = confirm("Are you sure you want to delete all data?");
    if (!confirmed) return;

    try {
        const res = await fetch("/api/data/delete-all", {
            method: "DELETE",
        });

        if (!res.ok) throw new Error("Failed to delete all data.");

        alert("All data deleted successfully.");
        await fetch_videos(); // Refresh video list after deletion
    } catch (err) {
        console.error(err);
        alert("Failed to delete all data.");
    }
}

function prev_page() {
    if (current_page.value > 1) current_page.value--;
}

function next_page() {
    if (current_page.value < total_pages.value) current_page.value++;
}

// Fetch the list of videos when the component is mounted
onMounted(() => {
    fetch_videos();
});
</script>

<template>
    <div class="view-container">
        <!-- Row for upload controls -->
        <div class="parameters-bar">
            <input type="file" @change="handle_file_change" />
            <div class="parameters-bar-group">
                <label for="sampling-fps-input">Sampling FPS:</label>
                <input id="sampling-fps-input" class="double-digit-input" type="number" step="0.1" v-model="sampling_fps" min="0.1" />
                <em>(Affects both upload and sync!)</em>
            </div>
            <div class="parameters-bar-group">
                <label for="bucket-only-checkbox">Bucket only:</label>
                <input id="bucket-only-checkbox" type="checkbox" v-model="bucket_only" />
            </div>
            <button @click="upload_video">Upload video</button>
        </div>

        <!-- Legend row -->
        <div class="video-list-legend">
            <div class="legend-item">Video</div>
            <div class="legend-item">Duration</div>
            <div class="legend-item">In bucket</div>
            <div class="legend-item">Embedding entries</div>
            <div class="legend-item">Actions</div>
        </div>

        <!-- Video list -->
        <div class="video-list">
            <div v-for="(video, index) in paginated_videos" :key="index" class="video-entry">
                <div class="video-entry-item">{{ video.video_name }}</div>
                <div class="video-entry-item">
                    {{ video.human_duration }} ({{ video.duration }} seconds)
                </div>
                <div class="video-entry-item">{{ video.in_bucket }}</div>
                <div class="video-entry-item">{{ video.embedding_entries }}</div>
                <div class="video-entry-item actions">
                    <button id="sync-video-button" @click="synchronize_video(video.video_name)">
                        Synchronize
                    </button>
                    <button id="delete-video-button" @click="delete_video(video.video_name)">
                        Delete
                    </button>
                </div>
            </div>
        </div>

        <!-- Pagination controls -->
        <div class="video-list-pagination" v-if="total_pages > 1">
            <button @click="prev_page" :disabled="current_page === 1">
                Previous
            </button>
            <span>Page {{ current_page }} of {{ total_pages }}</span>
            <button @click="next_page" :disabled="current_page === total_pages">
                Next
            </button>
        </div>

        <!-- Actions over all data -->
        <div class="parameters-bar" id="all-data-actions">
            <div class="parameters-bar-group">
                <label for="mirror-bucket-checkbox">Force bucket mirroring:</label>
                <input id="mirror-bucket-checkbox" type="checkbox" v-model="force_bucket_mirror" />
            </div>
            <button id="sync-video-button" @click="synchronize_all_data">
                Synchronize all data
            </button>
            <button id="delete-video-button" @click="delete_all_data">
                Delete all data
            </button>
        </div>
    </div>
</template>
