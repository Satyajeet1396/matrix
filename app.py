import streamlit as st
import numpy as np
import pandas as pd
import random
from typing import Tuple

def generate_integer_matrix_with_target_average(rows: int, cols: int, target_avg: float, 
                                                min_val: int = 0, max_val: int = 3) -> np.ndarray:
    total_elements = rows * cols
    target_sum = target_avg * total_elements
    matrix = np.full((rows, cols), min_val, dtype=int)
    current_sum = np.sum(matrix)
    remaining_sum = target_sum - current_sum

    positions = [(i, j) for i in range(rows) for j in range(cols)]
    random.shuffle(positions)

    pos_index = 0
    while remaining_sum > 0 and pos_index < len(positions):
        i, j = positions[pos_index % len(positions)]
        if matrix[i, j] < max_val and remaining_sum >= 1:
            increment = min(max_val - matrix[i, j], int(remaining_sum))
            matrix[i, j] += increment
            remaining_sum -= increment
        pos_index += 1
        if pos_index % total_elements == 0:
            random.shuffle(positions)

    max_passes = 10
    current_pass = 0
    while remaining_sum > 0.5 and current_pass < max_passes:
        random.shuffle(positions)
        for i, j in positions:
            if matrix[i, j] < max_val and remaining_sum > 0.5:
                matrix[i, j] += 1
                remaining_sum -= 1
                if remaining_sum <= 0.5:
                    break
        current_pass += 1
    return matrix

def calculate_possible_average_range(rows: int, cols: int, min_val: int = 0, max_val: int = 3) -> Tuple[float, float]:
    return min_val, max_val

def main():
    st.set_page_config(
        page_title="Integer Matrix Generator",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🔢 Integer Matrix Generator")
    st.markdown("Generate matrices with **integer values only** (0, 1, 2, 3) and custom average ranges")

    st.header("Matrix Parameters")
    col1, col2 = st.columns(2)
    with col1:
        rows = st.number_input("Number of Rows", min_value=1, max_value=20, value=4)
    with col2:
        cols = st.number_input("Number of Columns", min_value=1, max_value=20, value=16)

    min_possible, max_possible = calculate_possible_average_range(rows, cols)
    st.info(f"📊 **Possible Range:** {min_possible:.1f} - {max_possible:.1f}")

    col3, col4 = st.columns(2)
    with col3:
        min_avg_range = st.number_input("Minimum Average", min_value=min_possible, max_value=max_possible, value=2.0, step=0.1)
    with col4:
        max_avg_range = st.number_input("Maximum Average", min_value=min_avg_range, max_value=max_possible, value=2.5, step=0.1)

    if min_avg_range >= max_avg_range:
        st.error("⚠️ Maximum average must be greater than minimum average!")
        return

    if st.button("🎲 Generate New Matrix", type="primary"):
        target_avg = random.uniform(min_avg_range, max_avg_range)
        try:
            with st.spinner('Generating matrix...'):
                matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)
                st.session_state.matrix = matrix
                st.session_state.target_avg = target_avg
        except Exception as e:
            st.error(f"Error generating matrix: {e}")
            return

    if 'matrix' not in st.session_state:
        target_avg = random.uniform(min_avg_range, max_avg_range)
        matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)
        st.session_state.matrix = matrix
        st.session_state.target_avg = target_avg

    matrix = st.session_state.matrix
    target_avg = st.session_state.target_avg

    st.subheader("Generated Matrix")
    df = pd.DataFrame(matrix)
    df.columns = [f"Col {i+1}" for i in range(df.shape[1])]
    df.index = [f"Row {i+1}" for i in range(df.shape[0])]

    def color_matrix(val):
        if val == 0:
            return 'background-color: #ffebee; color: #c62828; font-weight: bold'
        elif val == 1:
            return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
        elif val == 2:
            return 'background-color: #f3e5f5; color: #7b1fa2; font-weight: bold'
        elif val == 3:
            return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
        return ''

    styled_df = df.style.applymap(color_matrix).format("{:.0f}")
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Matrix Summary")
    col1, col2, col3, col4 = st.columns(4)
    actual_avg = np.mean(matrix)
    with col1:
        st.metric("Target Avg", f"{target_avg:.3f}")
    with col2:
        st.metric("Actual Avg", f"{actual_avg:.3f}", delta=f"{actual_avg - target_avg:.3f}")
    with col3:
        st.metric("Min Value", f"{np.min(matrix)}")
    with col4:
        st.metric("Max Value", f"{np.max(matrix)}")

    st.subheader("Distribution")
    unique, counts = np.unique(matrix, return_counts=True)
    dist = dict(zip(unique, counts))
    total = rows * cols

    col5, col6 = st.columns(2)
    with col5:
        for i in range(4):
            count = dist.get(i, 0)
            percent = (count / total) * 100
            st.write(f"**{i}:** {count} ({percent:.1f}%)")

    with col6:
        bar_df = pd.DataFrame({
            'Value': ['0', '1', '2', '3'],
            'Count': [dist.get(i, 0) for i in range(4)]
        })
        st.bar_chart(bar_df.set_index("Value"))

    st.subheader("Export Matrix")
    csv = df.to_csv()
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"matrix_{rows}x{cols}_avg_{actual_avg:.2f}.csv",
        mime="text/csv"
    )

    with st.expander("🔎 Raw Matrix Data"):
        st.code(str(matrix))

    with st.expander("📘 How It Works"):
        st.markdown("""
        **Algorithm Steps:**
        1. Set all values to 0 initially.
        2. Compute how much sum is needed to reach the target average.
        3. Randomly pick positions and increase their value within the range.
        4. Repeat until the matrix sum approximates the desired target.
        5. All values are constrained between 0 and 3, and final result is integer-only.
        """)

if __name__ == "__main__":
    main()
