<template>
  <!-- Show layout only on authenticated routes -->
  <AppLayout v-if="isAuthenticated && $route.path !== '/login'">
    <router-view />
  </AppLayout>
  <router-view v-else />
</template>

<script setup>
import { computed } from 'vue'
import AppLayout from './components/AppLayout.vue'
import { useAuthStore } from './stores/auth.js'

const auth = useAuthStore()
const isAuthenticated = computed(() => auth.isAuthenticated)
</script>
