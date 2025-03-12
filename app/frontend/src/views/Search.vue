<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
const router = useRouter();

// Function for image retrieval from video
async function get_image_from_video(video_name, timestamp) {
    try {
        const response = await fetch(
            `/api/image-from-video/${video_name}?timestamp=${timestamp}`
        );
        if (!response.ok) {
            throw new Error(`HTTP error, status: ${response.status}`);
        }
        const blob = await response.blob();
        return URL.createObjectURL(blob);
    } catch (err) {
        console.error("Error fetching image from video", err);
    }
}

// Variables for the embedding search
const query = ref("");
const top_k = ref(5);
const results = ref([]);
const loading = ref(false);
const error = ref("");

// Function for the embedding search
async function search() {
    if (!query.value) return; // Don't search if query is empty

    loading.value = true;
    error.value = "";
    results.value = [];

    try {
        const response = await fetch(
            `/api/search-embeddings?text=${encodeURIComponent(query.value)}&top_k=${top_k.value
            }`
        );
        const data = await response.json();

        // Fetch images and add them to the search results
        const results_with_images = await Promise.all(
            data.results.map(async (item) => {
                const image_url = await get_image_from_video(
                    item.video_name,
                    item.timestamp
                );
                return { ...item, image_url };
            })
        );
        results.value = results_with_images;
    } catch (err) {
        error.value = "Error fetching results: " + err;
    } finally {
        loading.value = false;
        // If no results were found, show error
        if (!error.value && results.value.length === 0) {
            error.value =
                "No results found (the Milvus database collection is probably empty).";
        }
    }
}

// Function to redirect to analysis with specific search result
function startAnalysisHere(item) {
  const video_name = item.video_name;
  let start, end;
  if (item.timestamp < 5) {
    start = 0;
    end = 10;
  } else {
    start = item.timestamp - 5;
    end = item.timestamp + 5;
  }
  router.push({
    path: '/analysis',
    query: { video_name: video_name, start_seconds: start, end_seconds: end }
  });
}
</script>

<template>
    <div class="view-container">
        <!-- Search input parameters -->
        <div class="parameters-bar">
            <input type="text" id="search-text-input" v-model="query" placeholder="Enter search text (e.g., Black SUV)..."
                @keydown.enter="search" />
            <div class="parameters-bar-group">
                <label for="top-k-input">Top k results:</label>
                <input type="number" class="double-digit-input" id="top-k-input" v-model="top_k" min="1" max="50" />
            </div>
            <button @click="search">Search</button>
        </div>

        <!-- Search status (loading, error) -->
        <div class="search-status">
            <p v-if="loading" class="status">Loading search results...</p>
            <p v-if="error" class="error">{{ error }}</p>
        </div>

        <!-- Search results (images) -->
        <ul v-if="results.length > 0" class="search-results-grid">
            <li v-for="(item, index) in results" :key="index">
                <img class="search-image-result" :src="item.image_url" alt="Extracted frame" />
                <div class="search-item-details">
                    <p><strong>Video:</strong> {{ item.video_name }}</p>
                    <div class="search-item-details-middle-row">
                        <p><strong>Timestamp:</strong> {{ item.human_timestamp }} ({{ item.timestamp }} seconds)</p>
                        <button @click="startAnalysisHere(item)">Analyze this segment</button>
                    </div>
                    <p><strong>Similarity score:</strong> {{ item.similarity_score }}</p>
                </div>
            </li>
        </ul>
    </div>
</template>
