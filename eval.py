import os
import pandas as pd
import re

# Function to extract the option letter from the text
def extract_option_letter(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'Option:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1)
    else:
        return text[0].upper()


if __name__ == '__main__':
    # Path to the file containing the model's output results
    file_path = 'result/gpt-4o_output.csv'

    # Define the output directory
    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Apply the `extract_option_letter` function to the 'Output' column
    # and create a new column 'Extracted_Option' with the extracted letters
    df['Extracted_Option'] = df['Output'].apply(extract_option_letter)

    # Drop rows where 'Extracted_Option' is NaN (i.e., no valid option was extracted)
    df = df.dropna(subset=['Extracted_Option'])

    # Initialize a list to store accuracy data for each question category
    accuracy_data = []

    # Group the DataFrame by the 'question_category' column
    for category, group in df.groupby('question_category'):
        correct_count = (group['Extracted_Option'] == group['answer']).sum()
        accuracy = correct_count / len(group)
        accuracy_data.append({
            'Category': category,
            'Accuracy': accuracy,
            'Num': len(group)
        })

    # Calculate the overall accuracy across all categories
    total_correct = (df['Extracted_Option'] == df['answer']).sum()
    total_accuracy = total_correct / len(df)
    accuracy_data.append({
        'Category': 'Total',
        'Accuracy': total_accuracy
    })

    # Convert the accuracy data list into a DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # Construct the output file path for the Excel file
    output_file_name = file_path.replace('.csv', '.xlsx')
    output_file_name = output_file_name.replace('output', 'acc')
    output_excel_path = os.path.join(output_dir, os.path.basename(output_file_name))

    # Save the accuracy results to an Excel file
    with pd.ExcelWriter(output_excel_path) as writer:
        accuracy_df.to_excel(writer, index=False, sheet_name='Accuracy Results')

    # Print a message indicating the processing is complete
    print(f"Processed {file_path} and saved results to {output_excel_path}")
