import { createRouter, createWebHistory } from 'vue-router'
import Search from '../views/Search.vue'
import Analysis from '../views/Analysis.vue'
import Videos from '../views/Videos.vue'

const routes = [
  {
    // Redirect root URL to /search
    path: '/',
    redirect: '/search'
  },
  {
    path: '/search',
    name: 'Search',
    component: Search
  },
  {
    path: '/analysis',
    name: 'Analysis',
    component: Analysis
  },
  {
    path: '/videos',
    name: 'Videos',
    component: Videos
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
