import { createRouter, createWebHistory } from 'vue-router'
import LoginView from '../views/LoginView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/login' },
    { path: '/login', component: LoginView, meta: { public: true } },
  ],
})

// Auth guard — redirect to /login if no token
router.beforeEach((to) => {
  const token = localStorage.getItem('evs_token')
  if (!to.meta.public && !token) {
    return '/login'
  }
})

export default router
