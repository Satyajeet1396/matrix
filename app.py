import streamlit as st
import numpy as np
import pandas as pd
import random
from typing import Tuple

def generate_integer_matrix_with_target_average(rows: int, cols: int, target_avg: float, 
                                              min_val: int = 0, max_val: int = 3) -> np.ndarray:
    """
    Generate a matrix with only integer values (0, 1, 2, 3) 
    that achieves a target average as closely as possible.
    """
    total_elements = rows * cols
    target_sum = target_avg * total_elements

    # Start with all values at the minimum
    matrix = np.full((rows, cols), min_val, dtype=int)
    current_sum = np.sum(matrix)

    # Calculate how much we need to add to reach target sum
    remaining_sum = target_sum - current_sum

    # Create a list of all possible positions
    positions = [(i, j) for i in range(rows) for j in range(cols)]

    # Shuffle positions for randomness
    random.shuffle(positions)

    # Distribute the remaining sum by incrementing values
    pos_index = 0
    while remaining_sum > 0 and pos_index < len(positions):
        # Cycle through positions
        i, j = positions[pos_index % len(positions)]

        # Only increment if we can (within max_val limit) and need to
        if matrix[i, j] < max_val and remaining_sum >= 1:
            increment = min(max_val - matrix[i, j], int(remaining_sum))
            matrix[i, j] += increment
            remaining_sum -= increment

        pos_index += 1

        # If we've gone through all positions once, shuffle again for more randomness
        if pos_index % total_elements == 0:
            random.shuffle(positions)

    # If we still need to distribute more sum, do multiple passes with smaller increments
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
    """Calculate the theoretical minimum and maximum possible averages for given dimensions."""
    total_elements = rows * cols
    min_possible_avg = min_val
    max_possible_avg = max_val
    return min_possible_avg, max_possible_avg

def main():
    st.set_page_config(
        page_title="Integer Matrix Generator",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔢 Integer Matrix Generator")
    st.markdown("Generate matrices with **integer values only** (0, 1, 2, 3) and custom average ranges")

    # Sidebar for input parameters
    st.sidebar.header("Matrix Parameters")

    # Matrix dimensions
    rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=20, value=3, step=1)
    cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=20, value=3, step=1)

    # Calculate possible average range for current dimensions
    min_possible, max_possible = calculate_possible_average_range(rows, cols)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Average Range Settings")

    # Display possible range info
    st.sidebar.info(f"📊 **Possible Range**: {min_possible:.1f} - {max_possible:.1f}")

    # User-defined average range
    min_avg_range = st.sidebar.number_input(
        "Minimum Average", 
        min_value=float(min_possible), 
        max_value=float(max_possible), 
        value=1.5, 
        step=0.1,
        format="%.1f"
    )

    max_avg_range = st.sidebar.number_input(
        "Maximum Average", 
        min_value=min_avg_range, 
        max_value=float(max_possible), 
        value=2.5, 
        step=0.1,
        format="%.1f"
    )

    # Validate range
    if min_avg_range >= max_avg_range:
        st.sidebar.error("⚠️ Maximum average must be greater than minimum average!")
        return

    # Generate random target average within user range
    if st.sidebar.button("🎲 Generate New Matrix", type="primary", key="generate_btn"):
        # Generate random target average within specified range
        target_avg = random.uniform(min_avg_range, max_avg_range)

        try:
            with st.spinner('Generating integer matrix...'):
                matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)

                # Store in session state
                st.session_state.matrix = matrix
                st.session_state.target_avg = target_avg
                st.session_state.rows = rows
                st.session_state.cols = cols
                st.session_state.min_avg_range = min_avg_range
                st.session_state.max_avg_range = max_avg_range

        except Exception as e:
            st.error(f"Error generating matrix: {str(e)}")
            return

    # Initialize with default matrix if none exists
    if 'matrix' not in st.session_state:
        target_avg = random.uniform(min_avg_range, max_avg_range)
        matrix = generate_integer_matrix_with_target_average(rows, cols, target_avg)
        st.session_state.matrix = matrix
        st.session_state.target_avg = target_avg
        st.session_state.rows = rows
        st.session_state.cols = cols
        st.session_state.min_avg_range = min_avg_range
        st.session_state.max_avg_range = max_avg_range

    # Display results
    if 'matrix' in st.session_state:
        matrix = st.session_state.matrix
        target_avg = st.session_state.target_avg

        # Create layout columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Generated Integer Matrix")

            # Display matrix as a styled dataframe
            df = pd.DataFrame(matrix)
            df.columns = [f"Col {i+1}" for i in range(df.shape[1])]
            df.index = [f"Row {i+1}" for i in range(df.shape[0])]

            # Style the dataframe with integer formatting
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

            # Show the actual matrix values in a more compact way
            st.subheader("Matrix Values (Grid View)")
            matrix_display = ""
            for i in range(matrix.shape[0]):
                row_str = " | ".join([f"{matrix[i,j]:1.0f}" for j in range(matrix.shape[1])])
                matrix_display += f"**Row {i+1}:** {row_str}\n\n"
            st.markdown(matrix_display)

        with col2:
            st.subheader("Matrix Statistics")

            # Calculate statistics
            actual_avg = np.mean(matrix)
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            std_dev = np.std(matrix)

            # Display target vs actual
            st.metric("🎯 Target Average", f"{target_avg:.3f}")
            st.metric("📊 Actual Average", f"{actual_avg:.3f}", 
                     delta=f"{actual_avg - target_avg:.3f}")

            st.metric("📉 Minimum Value", f"{min_val}")
            st.metric("📈 Maximum Value", f"{max_val}")
            st.metric("📏 Standard Deviation", f"{std_dev:.3f}")

            # Validation indicators
            st.subheader("Validation Status")

            # Check if all values are integers in range
            all_integers = np.all((matrix >= 0) & (matrix <= 3) & (matrix == matrix.astype(int)))
            if all_integers:
                st.success("✅ All values are integers (0, 1, 2, 3)")
            else:
                st.error("❌ Some values are not valid integers")

            # Check if average is in user-specified range
            avg_in_range = min_avg_range <= actual_avg <= max_avg_range
            if avg_in_range:
                st.success(f"✅ Average is within specified range ({min_avg_range:.1f} - {max_avg_range:.1f})")
            else:
                st.warning(f"⚠️ Average slightly outside range ({min_avg_range:.1f} - {max_avg_range:.1f})")

            # Show how close we got to target
            deviation = abs(actual_avg - target_avg)
            if deviation < 0.1:
                st.success(f"✅ Very close to target! (±{deviation:.3f})")
            elif deviation < 0.2:
                st.info(f"ℹ️ Close to target (±{deviation:.3f})")
            else:
                st.warning(f"⚠️ Target deviation: ±{deviation:.3f}")

        # Additional information
        st.subheader("Generation Details")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:
            st.info(f"**Dimensions:** {rows} × {cols}")

        with info_col2:
            st.info(f"**Total Elements:** {rows * cols}")

        with info_col3:
            st.info(f"**Matrix Sum:** {np.sum(matrix)}")

        with info_col4:
            st.info(f"**Value Range:** 0-3 (integers)")

        # Value distribution
        st.subheader("Value Distribution")
        dist_col1, dist_col2 = st.columns([1, 1])

        with dist_col1:
            # Count occurrences of each value
            unique, counts = np.unique(matrix, return_counts=True)
            for val, count in zip(unique, counts):
                percentage = (count / (rows * cols)) * 100
                st.write(f"**Value {val}:** {count} times ({percentage:.1f}%)")

        with dist_col2:
            # Create a simple bar chart data
            distribution_data = {
                'Value': ['0', '1', '2', '3'],
                'Count': [0, 0, 0, 0]
            }

            for val, count in zip(unique, counts):
                distribution_data['Count'][int(val)] = count

            dist_df = pd.DataFrame(distribution_data)
            st.bar_chart(dist_df.set_index('Value'))

        # Export functionality
        st.subheader("Export Matrix")

        # Convert matrix to CSV
        csv = df.to_csv()
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"integer_matrix_{rows}x{cols}_avg_{actual_avg:.2f}.csv",
            mime="text/csv"
        )

        # Raw matrix display (expandable)
        with st.expander("View Raw Matrix Data"):
            st.code(str(matrix))

        # Algorithm explanation
        with st.expander("How the Algorithm Works"):
            st.markdown("""
            **Integer Matrix Generation Algorithm:**

            1. 🎯 **Target Selection**: Randomly select target average within your specified range
            2. 🔢 **Initialize**: Start with all values at minimum (0)
            3. 📊 **Calculate Need**: Determine total sum needed for target average
            4. 🎲 **Random Distribution**: Randomly distribute increments across matrix positions
            5. 🔄 **Iterative Adjustment**: Make multiple passes to get as close as possible to target
            6. ✅ **Validation**: Ensure all values remain integers between 0-3

            **Note**: Due to integer constraints, the actual average may vary slightly from the target.
            The algorithm attempts to get as close as possible while maintaining integer values only.
            """)

if __name__ == "__main__":
    main()
