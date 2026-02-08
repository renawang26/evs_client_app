# EVS样式及颜色定义

# 颜色图例
COLOR_LEGEND = {
    "pair_type_S": "#FFB6C1",  # 浅粉红色，用于EVS起点标记
    "pair_type_E": "#90EE90",  # 浅绿色，用于EVS终点标记
    "annotate_T": "#FFD700",   # 金色，用于标注
    "default": "#FFFFFF"       # 白色，默认背景
}

# 背景颜色样式常量
GREEN = 'background-color: #90EE90'
YELLOW = 'background-color: #FFD700'
PINK = 'background-color: #FFB6C1'

# EVS图例说明
EVS_LEGEND = """
<div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>
  <p><strong>EVS配对颜色说明：</strong></p>
  <ul>
    <li><span style='background-color:#FFB6C1; padding:2px 5px;'>起点</span> - 英文单词 (S)</li>
    <li><span style='background-color:#90EE90; padding:2px 5px;'>终点</span> - 中文单词 (E)</li>
    <li><span style='background-color:#FFD700; padding:2px 5px;'>标注</span> - 已标记 (T)</li>
  </ul>
</div>
"""


COLOR_LEGEND_legend = """
<div style="display: inline; float: right; font-size:14px; font-weight:bold;">
    <span style="font-weight:bold;">Fluency Legend</span>
    <span style="background-color:#C4E1FF; padding: 5px; 10px 5px 10px;">Pause</span>
    <span style="color:white; background-color:#336666; padding: 5px; 10px 5px 10px;">FP</span>
    <span style="color:white; background-color:#F75000; padding: 5px; 10px 5px 10px;">RLUP</span>
    <span style="color:white; background-color:#A23400; padding: 5px; 10px 5px 10px;">PLUP</span>
    <span style="color:white; background-color:#003060; padding: 5px; 10px 5px 10px;">Interpret</span>
    <span style="color:white; background-color:#977C00; padding: 5px; 10px 5px 10px;">PSA</span>
    <span style="color:C4E1FF; background-color:#EAC100; padding: 5px; 10px 5px 10px;">RSA</span>
</div>
"""

EVS_LEGEND_legend = """
<div style="display: flex; align-items: center; gap: 8px;">
    <span style="font-weight: bold;">EVS Legend</span>
    <span style="background-color: #FFD700; padding: 5px 10px; border-radius: 3px;">Pair</span>
    <span style="background-color: #9ACD32; padding: 5px 10px; border-radius: 3px; color: white;">Annotate</span>
    <span style="background-color: #00CED1; padding: 5px 10px; border-radius: 3px; color: white;">Pair & Annotate</span>
</div>
"""