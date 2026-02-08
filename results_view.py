import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from db_utils import EVSDataUtils

# Resource paths
EVS_RESOURCES_PATH = r"./evs_resources"

def render_whisper_results_tab():
    """显示 Whisper ASR 转录结果的标签页"""
    st.subheader("Whisper ASR 转录结果")

    # 获取所有已转录的文件
    files_df = EVSDataUtils.get_whisper_asr_files()
    
    if files_df.empty:
        st.info("暂无转录结果。请先在 'OpenAI Whisper' 标签页中转录音频文件。")
        return

    # 选择文件
    selected_file = st.selectbox(
        "选择文件",
        options=files_df['file_name'].unique(),
        key='whisper_results_file'
    )

    if selected_file:
        # 获取文件的切片信息
        slices_df = EVSDataUtils.get_whisper_asr_slices(selected_file)
        
        # 显示文件信息
        st.markdown("### 文件信息")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总切片数", len(slices_df))
        with col2:
            total_duration = slices_df['end_time'].max() - slices_df['start_time'].min()
            st.metric("总时长", f"{total_duration:.2f}秒")

        # 创建切片选择器
        slice_numbers = sorted(slices_df['slice_number'].unique())
        
        def format_slice_option(x):
            slice_data = slices_df[slices_df['slice_number'] == x]
            if slice_data.empty:
                return f"切片 {x}"
            start_time = slice_data['start_time'].iloc[0]
            end_time = slice_data['end_time'].iloc[0]
            return f"切片 {x} ({start_time:.1f}s - {end_time:.1f}s)"
            
        selected_slice = st.selectbox(
            "选择切片",
            options=slice_numbers,
            format_func=format_slice_option,
            key='whisper_results_slice'
        )

        if selected_slice is not None:
            # 获取选定切片的转录结果
            words_df, segments_df = EVSDataUtils.get_whisper_asr_results(
                file_name=selected_file,
                slice_number=selected_slice
            )

            # 显示音频播放器
            audio_path = os.path.join(
                EVS_RESOURCES_PATH,
                "slice_audio_files",
                os.path.splitext(selected_file)[0],
                f"{selected_slice}.mp3"
            )
            if os.path.exists(audio_path):
                st.audio(audio_path)

            # 创建中英文标签页
            lang_tabs = st.tabs(["中文", "英文"])
            
            # 处理中文结果
            with lang_tabs[0]:
                zh_words = words_df[words_df['lang'] == 'zh'].copy()
                zh_segments = segments_df[segments_df['lang'] == 'zh'].copy()
                
                if not zh_words.empty:
                    st.markdown("#### 中文转录结果")
                    
                    # 显示段落级别的转录
                    st.markdown("##### 段落级别")
                    segments_display = zh_segments[['start_time', 'end_time', 'text']].copy()
                    segments_display.columns = ['开始时间', '结束时间', '文本']
                    st.dataframe(segments_display)
                    
                    # 显示词级别的转录
                    st.markdown("##### 词级别")
                    words_display = zh_words[['start_time', 'end_time', 'word', 'confidence']].copy()
                    words_display.columns = ['开始时间', '结束时间', '词', '置信度']
                    st.dataframe(words_display)
                    
                    # 显示统计信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总词数", len(zh_words))
                    with col2:
                        avg_confidence = zh_words['confidence'].mean()
                        st.metric("平均置信度", f"{avg_confidence:.2%}")
                    with col3:
                        speech_duration = zh_words['end_time'].max() - zh_words['start_time'].min()
                        st.metric("说话时长", f"{speech_duration:.2f}秒")
                else:
                    st.info("该切片没有中文转录结果")
            
            # 处理英文结果
            with lang_tabs[1]:
                en_words = words_df[words_df['lang'] == 'en'].copy()
                en_segments = segments_df[segments_df['lang'] == 'en'].copy()
                
                if not en_words.empty:
                    st.markdown("#### English Transcription Results")
                    
                    # 显示段落级别的转录
                    st.markdown("##### Segment Level")
                    segments_display = en_segments[['start_time', 'end_time', 'text']].copy()
                    segments_display.columns = ['Start Time', 'End Time', 'Text']
                    st.dataframe(segments_display)
                    
                    # 显示词级别的转录
                    st.markdown("##### Word Level")
                    words_display = en_words[['start_time', 'end_time', 'word', 'confidence']].copy()
                    words_display.columns = ['Start Time', 'End Time', 'Word', 'Confidence']
                    st.dataframe(words_display)
                    
                    # 显示统计信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Words", len(en_words))
                    with col2:
                        avg_confidence = en_words['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    with col3:
                        speech_duration = en_words['end_time'].max() - en_words['start_time'].min()
                        st.metric("Speech Duration", f"{speech_duration:.2f}s")
                else:
                    st.info("No English transcription results for this slice")

            # 显示可视化
            st.markdown("### 时序分析")
            
            # 准备数据
            plot_data = []
            for _, row in words_df.iterrows():
                plot_data.append({
                    'word': row['word'],
                    'start': row['start_time'],
                    'end': row['end_time'],
                    'lang': '中文' if row['lang'] == 'zh' else 'English',
                    'confidence': row['confidence']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            if not plot_df.empty:
                # 创建时序图
                fig = go.Figure()
                
                # 添加中文轨道
                zh_data = plot_df[plot_df['lang'] == '中文']
                if not zh_data.empty:
                    fig.add_trace(go.Scatter(
                        x=zh_data['start'],
                        y=[1] * len(zh_data),
                        mode='markers+text',
                        name='中文',
                        text=zh_data['word'],
                        textposition='top center',
                        marker=dict(
                            size=10,
                            color=zh_data['confidence'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title='置信度')
                        )
                    ))
                
                # 添加英文轨道
                en_data = plot_df[plot_df['lang'] == 'English']
                if not en_data.empty:
                    fig.add_trace(go.Scatter(
                        x=en_data['start'],
                        y=[0] * len(en_data),
                        mode='markers+text',
                        name='English',
                        text=en_data['word'],
                        textposition='bottom center',
                        marker=dict(
                            size=10,
                            color=en_data['confidence'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title='Confidence')
                        )
                    ))
                
                # 更新布局
                fig.update_layout(
                    title='双语时序对照图',
                    xaxis_title='时间 (秒)',
                    yaxis=dict(
                        ticktext=['English', '中文'],
                        tickvals=[0, 1],
                        range=[-1, 2]
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("没有足够的数据生成可视化图表") 