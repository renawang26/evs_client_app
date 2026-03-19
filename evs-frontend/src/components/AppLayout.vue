<template>
  <el-container class="layout">
    <el-aside width="200px" class="sidebar">
      <div class="logo">EVS Navigation</div>
      <el-menu :router="true" :default-active="$route.path">
        <el-menu-item index="/files">
          <el-icon><Document /></el-icon>
          <span>Files</span>
        </el-menu-item>
        <el-menu-item index="/transcribe">
          <el-icon><Microphone /></el-icon>
          <span>Transcribe</span>
        </el-menu-item>
        <el-menu-item index="/annotate">
          <el-icon><EditPen /></el-icon>
          <span>Annotate</span>
        </el-menu-item>
      </el-menu>
      <div class="sidebar-footer">
        <el-button text @click="handleLogout" size="small">
          <el-icon><SwitchButton /></el-icon> Logout
        </el-button>
      </div>
    </el-aside>

    <el-main class="main-content">
      <slot />
    </el-main>
  </el-container>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth.js'
import { Document, Microphone, EditPen, SwitchButton } from '@element-plus/icons-vue'

const router = useRouter()
const auth = useAuthStore()

function handleLogout() {
  auth.logout()
  router.push('/login')
}
</script>

<style scoped>
.layout { height: 100vh; }
.sidebar { background: #304156; display: flex; flex-direction: column; }
.logo {
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  padding: 20px 16px;
  border-bottom: 1px solid #404c5e;
}
.sidebar :deep(.el-menu) {
  background: #304156;
  border-right: none;
  flex: 1;
}
.sidebar :deep(.el-menu-item) { color: #bfcbd9; }
.sidebar :deep(.el-menu-item.is-active) { color: #fff; background: #263445; }
.sidebar :deep(.el-menu-item:hover) { background: #263445; }
.sidebar-footer { padding: 12px; border-top: 1px solid #404c5e; }
.sidebar-footer .el-button { color: #bfcbd9; }
.main-content { padding: 0; background: #f0f2f5; overflow-y: auto; }
</style>
