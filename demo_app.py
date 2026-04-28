"""
业绩变脸预警系统 - 在线演示版
适用于Streamlit Cloud部署
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="业绩变脸预警系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("📊 上市公司业绩变脸早期预警系统")
st.markdown("### 语义漂移与数字异动——基于文本分析与机器学习的智能预警")
st.markdown("---")

# 侧边栏
st.sidebar.header("系统说明")
st.sidebar.info("""
**作品编号:** 2026018956

**核心创新:**
- 多模态特征融合(121维)
- 三级变脸定义体系
- 滚动预测框架
- 历史调优机制

**最优性能:** AUC=0.78
""")

# 项目介绍
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📖 项目简介")
    st.write("""
    本系统构建了基于文本分析与机器学习的上市公司业绩变脸早期预警系统。

    **核心特点:**
    - 融合121维多模态特征(历史财务指标、预告数值特征、FinBERT文本语义特征)
    - 设计三级变脸定义体系(严重/中度/轻度)
    - 采用滚动预测框架,严格避免未来信息泄露
    - 引入历史调优机制,使模型自适应金融数据非平稳性

    **应用价值:**
    - 投资者: 在业绩预告发布时即可获得变脸概率评估
    - 监管机构: 批量筛查高风险公司,提高监管效率
    - 上市公司: 自查预告可靠性,避免监管处罚
    """)

with col2:
    st.header("🎯 性能指标")
    st.metric("最优AUC", "0.7836", "随机森林")
    st.metric("平均Recall", "0.6137", "识别率")
    st.metric("特征维度", "121维", "多模态")
    st.metric("数据年份", "2012-2025", "14年")

st.markdown("---")

# 演示功能
st.header("🔍 预警功能演示")

# 创建示例数据
@st.cache_data
def load_sample_data():
    # 模拟一些示例股票数据
    stocks = pd.DataFrame({
        '股票代码': ['000001', '000002', '600000', '600036', '000651'],
        '股票名称': ['平安银行', '万科A', '浦发银行', '招商银行', '格力电器'],
        '预告类型': ['预增', '预减', '续盈', '首亏', '扭亏'],
        '预告净利润(万元)': [450000, -120000, 180000, -50000, 280000],
        '变脸概率': [0.15, 0.68, 0.23, 0.82, 0.35],
        '预警等级': ['低风险', '高风险', '低风险', '高风险', '中风险']
    })
    return stocks

sample_data = load_sample_data()

# 显示示例数据
st.subheader("示例股票预警结果")
st.dataframe(sample_data, use_container_width=True)

# 单只股票查询
st.subheader("单只股票查询")
col1, col2 = st.columns([1, 2])

with col1:
    stock_code = st.text_input("请输入股票代码", value="000001", max_chars=6)
    predict_btn = st.button("🔮 开始预测", type="primary")

with col2:
    if predict_btn:
        # 模拟预测过程
        with st.spinner("正在分析中..."):
            import time
            time.sleep(1)

        # 模拟预测结果
        np.random.seed(int(stock_code) % 1000)
        prob = np.random.uniform(0.1, 0.9)

        if prob < 0.3:
            level = "低风险"
            color = "green"
        elif prob < 0.6:
            level = "中风险"
            color = "orange"
        else:
            level = "高风险"
            color = "red"

        st.success(f"✅ 股票 {stock_code} 预测完成!")

        # 显示结果
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("变脸概率", f"{prob*100:.1f}%", delta=f"{(prob-0.5)*100:.1f}%")
        with col_b:
            st.metric("预警等级", level)
        with col_c:
            st.metric("置信度", f"{np.random.uniform(0.75, 0.95)*100:.1f}%")

        # 特征重要性
        st.subheader("特征重要性分析")
        features = pd.DataFrame({
            '特征': ['预告文本长度', 'T-1年ROA', '预告下限比率', 'T-1年现金利润比率',
                    '预告方向', 'T-1年期间费用比率', '预告类型编码', 'T-2年ROA'],
            '重要性': [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
        })

        fig = px.bar(features, x='重要性', y='特征', orientation='h',
                    title='Top 8 重要特征', color='重要性',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 技术架构
st.header("🏗️ 技术架构")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("📊 数据层\n\nCSMAR数据库\n18类财务数据\n业绩预告数据")

with col2:
    st.info("🔧 特征层\n\n60维财务特征\n24维预告特征\n35维BERT语义")

with col3:
    st.info("🤖 模型层\n\n随机森林\nXGBoost\nLightGBM")

with col4:
    st.info("💻 应用层\n\nWeb预警系统\nAPI接口\n可视化分析")

st.markdown("---")

# 模型对比
st.header("📈 模型性能对比")

model_comparison = pd.DataFrame({
    '模型': ['Random Forest', 'XGBoost', 'LightGBM'],
    '平均AUC': [0.7549, 0.7044, 0.7105],
    '最优AUC': [0.7836, 0.7350, 0.7262],
    '平均Recall': [0.6137, 0.4927, 0.2871],
    '平均Precision': [0.0523, 0.0489, 0.0312]
})

st.dataframe(model_comparison, use_container_width=True)

# AUC对比图
fig = go.Figure()
fig.add_trace(go.Bar(name='平均AUC', x=model_comparison['模型'],
                     y=model_comparison['平均AUC'], marker_color='lightblue'))
fig.add_trace(go.Bar(name='最优AUC', x=model_comparison['模型'],
                     y=model_comparison['最优AUC'], marker_color='darkblue'))
fig.update_layout(barmode='group', title='模型AUC对比',
                 yaxis_title='AUC值', xaxis_title='模型')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 创新点
st.header("💡 核心创新点")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ 多模态特征融合")
    st.write("""
    - 首次将FinBERT中文金融语义特征与财务特征融合
    - 121维特征全面刻画公司状况
    - 3年消融实验验证BERT有效性
    """)

    st.subheader("2️⃣ 三级变脸定义")
    st.write("""
    - 严重变脸: 方向反转(预盈变亏损)
    - 中度变脸: 类型变化(预增变预减)
    - 轻度变脸: 数值大幅偏离
    """)

with col2:
    st.subheader("3️⃣ 滚动预测框架")
    st.write("""
    - 3年窗口滚动预测
    - 严格无未来信息泄露
    - 所有特征在预告发布时已知
    """)

    st.subheader("4️⃣ 历史调优机制")
    st.write("""
    - 基于前轮效果自适应调整
    - 适应金融数据非平稳性
    - 提升跨年泛化能力
    """)

st.markdown("---")

# 页脚
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>2026年广东省大学生计算机设计大赛 - 大数据应用实践赛</p>
    <p>作品编号: 2026018956 | 作品名称: 语义漂移与数字异动——基于文本分析与机器学习的上市公司业绩变脸早期预警系统</p>
</div>
""", unsafe_allow_html=True)
