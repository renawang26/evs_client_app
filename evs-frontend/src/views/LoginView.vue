<template>
  <div class="login-container">
    <el-card class="login-card">
      <template #header>
        <h2>EVS Navigation System</h2>
      </template>

      <el-form :model="form" @submit.prevent="handleLogin" label-width="80px">
        <el-form-item label="Email">
          <el-input v-model="form.email" type="email" placeholder="user@example.com" />
        </el-form-item>
        <el-form-item label="Password">
          <el-input v-model="form.password" type="password" placeholder="Password" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" native-type="submit" :loading="loading" block>
            Login
          </el-button>
        </el-form-item>
        <el-alert v-if="error" :title="error" type="error" show-icon />
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth.js'

const router = useRouter()
const auth = useAuthStore()
const form = ref({ email: '', password: '' })
const loading = ref(false)
const error = ref(null)

async function handleLogin() {
  loading.value = true
  error.value = null
  try {
    await auth.login(form.value.email, form.value.password)
    router.push('/files')
  } catch {
    error.value = 'Invalid email or password'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: #f0f2f5;
}
.login-card {
  width: 400px;
}
.login-card h2 {
  margin: 0;
  font-size: 18px;
}
</style>
