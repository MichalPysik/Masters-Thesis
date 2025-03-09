<script setup>
</script>


<template>
    <div class="app-container">
        <!--- Top bar containing navigation and header -->
        <div class ="top-bar">
            <nav>
                <router-link to="/search">Search</router-link>
                <router-link to="/analysis">Analysis</router-link>
                <router-link to="/videos">Videos</router-link>
            </nav>
            <header class="page-header">
                <h1>Traffic footage system</h1>
            </header>
        </div>
        
        <!--- The routed view (Search, Analysis, Videos)-->    
        <router-view />
    </div>
</template>
