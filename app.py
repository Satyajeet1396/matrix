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

def main():
    st.set_page_config(page_title="Integer Matrix Generator", layout="wide")

    st.title("🔢 Integer Matrix Generator")

    col1, col2, col3 = st.columns(3)
    rows = col1.number_input("Rows", min_value=1, max_value=20, value=4, step=1)
    cols = col2.number_input("Columns", min_value=1, max_value=20, value=16, step=1)
    generate = col3.button("🎲 Generate Matrix")

    col4, col5 = st.columns(2)
    min_avg = col4.number_input("Min Avg", min_value=0.0, max_value=3.0, value=2.0, step=0.1, format="%.1f")
    max_avg = col5.number_input("Max Avg", min_value=min_avg, max_value=3.0, value=2.5, step=0.1, format="%.1f")

    if generate:
        target_avg = random.uniform(min_avg, max_avg)
        matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)
        st.session_state.matrix = matrix
        st.session_state.target_avg = target_avg

    if "matrix" in st.session_state:
        matrix = st.session_state.matrix
        target_avg = st.session_state.target_avg
        actual_avg = np.mean(matrix)

        st.markdown("## 🎯 Matrix Output")
        df = pd.DataFrame(matrix)

        def highlight_cells(val):
            color_map = {
                0: 'background-color: #fce4ec; color: black; font-size: 24px; text-align: center;',
                1: 'background-color: #fff9c4; color: black; font-size: 24px; text-align: center;',
                2: 'background-color: #e1f5fe; color: black; font-size: 24px; text-align: center;',
                3: 'background-color: #c8e6c9; color: black; font-size: 24px; text-align: center;',
            }
            return color_map.get(val, '')

        styled_df = df.style.applymap(highlight_cells)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Avg", f"{target_avg:.3f}")
        col2.metric("Actual Avg", f"{actual_avg:.3f}")
        col3.metric("Std Dev", f"{np.std(matrix):.3f}")

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name="matrix.csv", mime="text/csv")

if __name__ == "__main__":
    main()
