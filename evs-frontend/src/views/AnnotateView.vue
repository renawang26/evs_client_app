<template>
  <div class="annotate-view">
    <el-page-header title="EVS Annotation" @back="$router.push('/files')" />

    <el-card style="margin-top: 16px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>EVS Word-Level Annotation</span>
          <el-select
            v-model="selectedFileId"
            placeholder="Select file to annotate"
            style="width: 280px"
            @change="loadAnnotator"
          >
            <el-option
              v-for="file in filesStore.files"
              :key="file.id"
              :label="file.file_name"
              :value="file.id"
            />
          </el-select>
        </div>
      </template>

      <div v-if="!selectedFileId" class="empty-state">
        <el-empty description="Select a transcribed file to start annotation" />
      </div>

      <iframe
        v-else
        :src="annotatorSrc"
        class="annotator-frame"
        sandbox="allow-scripts allow-same-origin"
        title="EVS Annotator"
      />
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useFilesStore } from '../stores/files.js'

const filesStore = useFilesStore()
const selectedFileId = ref(null)

// Serve the existing annotator from the static path
// FastAPI serves evs_annotator/ at /static/annotator/
const annotatorSrc = computed(() =>
  selectedFileId.value
    ? `/static/annotator/index.html?file_id=${selectedFileId.value}`
    : null
)

function loadAnnotator(fileId) {
  selectedFileId.value = fileId
}

onMounted(() => filesStore.fetchFiles())
</script>

<style scoped>
.annotate-view { padding: 24px; }
.annotator-frame {
  width: 100%;
  height: 600px;
  border: none;
}
.empty-state { padding: 40px 0; }
</style>
