<script setup>
</script>


<template>
    <div class="app-container">
        <header class="page-header">
            <h1>System for search and analysis of traffic footage</h1>
        </header>

        <nav>
            <router-link to="/search">Search</router-link>
            <router-link to="/analysis">Analysis</router-link>
            <router-link to="/videos">Videos</router-link>
        </nav>
  
        <!--- The routed view (Search, Analysis, Videos)-->    
        <router-view />
    </div>
</template>
