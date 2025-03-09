<script setup>
import { ref } from "vue";

// Function for image retrieval from video
async function get_image_from_video(video_name, timestamp) {
    try {
        const response = await fetch(`/api/image-from-video/${video_name}?timestamp=${timestamp}`);
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
const search_query = ref("");
const search_top_k = ref(5);    
const search_results = ref([]); 
const search_loading = ref(false);
const search_error = ref("");

// Function for the embedding search
async function search() {
    if (!search_query.value) return; // Don't search if query is empty

    search_loading.value = true;
    search_error.value = "";
    search_results.value = [];

    try {
        const response = await fetch(`/api/search-embeddings?text=${encodeURIComponent(search_query.value)}&top_k=${search_top_k.value}`);
        const data = await response.json();

        // Fetch images and add them to the search results
        const results_with_images = await Promise.all(
            data.results.map(async (item) => {
                const image_url = await get_image_from_video(item.video_name, item.timestamp);
                return { ...item, image_url };
            })
        );
        search_results.value = results_with_images;
    } catch (err) {
        search_error.value = "Error fetching results";
    } finally {
        search_loading.value = false;
    }
}
</script>


<template>
    <div class="view-container">
        <div class="search-bar">
            <input type="text" v-model="search_query" placeholder="Enter search text (e.g., Orange SUV)..." @keydown.enter="search"/>
            <label for="top-k-input">Top k results:</label>
            <input type="number" id="top-k-input" v-model="search_top_k" min="1" max="50" />
            <button @click="search">Search</button>
        </div>

        <p v-if="search_loading" class="status">Loading search results...</p>
        <p v-if="search_error" class="error">{{ search_error }}</p>

        <ul v-if="search_results.length > 0" class="results-grid">
            <li v-for="(item, index) in search_results" :key="index">
                <img class="search-image-result" :src="item.image_url" alt="Extracted frame" />
                <div class="item-details">
                    <p><strong>Video:</strong> {{ item.video_name }}</p>
                    <p><strong>Timestamp:</strong> {{ item.human_timestamp }} ({{ item.timestamp }} seconds)</p>
                    <p><strong>Similarity score:</strong> {{ item.similarity_score }}</p>
                </div>
            </li>
        </ul>
        <p v-else>No search results found.</p>
    </div>
</template>


