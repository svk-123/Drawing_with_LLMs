import google.generativeai as genai
import asyncio
import os
import pandas as pd
# from tqdm.auto import tqdm # Still tricky with async, using print statements for now

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

# Initialize the model once
MODEL_NAME = "gemini-2.0-flash" # Using 1.5 flash for async support
model = genai.GenerativeModel(MODEL_NAME)


# --- Async Function to get SVG code from Gemini (Same as before) ---
async def get_svg_code_gemini_async(instruction_text):
    """
    Generates SVG code asynchronously from a text instruction using the Gemini API.
    Returns the response text and any error encountered.
    """
    instruction = f"""
            Generate SVG code to visually represent the following text description, while respecting the given constraints.
            <constraints>
            * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
            * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
            </constraints>

            Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints.
            Focus on a clear and concise representation of the input description within the given limitations.
            Always give the complete SVG code with nothing omitted. Never use an ellipsis.

            The code is scored based on similarity to the description, Visual question anwering and aesthetic components.
            Please generate a detailed svg code accordingly.

            input description: {instruction_text}
            """

    try:
        response = await model.generate_content_async(
            contents=instruction,
             generation_config=genai.types.GenerationConfig(
                 temperature=0.4,
                 # Consider max_output_tokens for SVG output
             )
        )

        if response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text, None
        else:
             print(f"Warning: No candidates returned for instruction starting '{instruction_text[:50]}...'")
             return None, "No content candidates generated"

    except Exception as e:
        print(f"Error processing instruction starting '{instruction_text[:50]}...': {e}")
        return None, str(e)


# --- Async function to process a batch of instructions (Same as before) ---
async def process_batch_async(batch_df):
    """
    Processes a pandas DataFrame batch asynchronously using get_svg_code_gemini_async.
    Adds 'gemini_response_text' and 'gemini_error' columns to the batch DataFrame.
    """
    tasks = []
    for _, row in batch_df.iterrows():
        instruction_text = row["description"]
        tasks.append(asyncio.create_task(get_svg_code_gemini_async(instruction_text)))

    print(f"Executing {len(tasks)} async tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("All async tasks in batch completed.")

    response_texts = []
    errors = []
    for res in results:
        if isinstance(res, Exception):
            response_texts.append(None)
            errors.append(str(res))
        else:
            text, error = res
            response_texts.append(text)
            errors.append(error)

    batch_df["gemini_response_text"] = response_texts
    batch_df["gemini_error"] = errors

    return batch_df

# --- Main Async Batch Processor ---
async def main_batch_processor():
    """
    Main async function to load data, iterate through batches,
    process each batch asynchronously, and save results.
    """
    df = pd.read_csv('./drawing-with-llms/description_master_gemini_20_48k.csv')
    df = df.iloc[20000:] # Use a smaller sample for testing

    batch_size = 100
    total_rows = len(df)

    os.makedirs("batches_async", exist_ok=True)

    print(f"Total rows to process: {total_rows}")

    # Iterate through batches - This loop is now inside an async function
    for i in range(0, total_rows, batch_size):
        batch_num = i // batch_size + 1
        start_idx = i
        end_idx = min(i + batch_size, total_rows)
        print(f"\n--- Processing batch {batch_num} ({start_idx}-{end_idx-1}) ---")

        filename = f"batches_async/response_batch_{batch_num}.csv"

        if os.path.exists(filename):
            print(f"Skipping batch {batch_num}, already exists.")
            continue

        batch_df_slice = df.iloc[start_idx:end_idx].copy()

        # Await the async processing function for the batch
        # Since we are already in an async function, we use 'await'
        # instead of 'asyncio.run()'
        processed_batch_df = await process_batch_async(batch_df_slice)

        # Save the processed batch (this part remains synchronous as file I/O is sync here)
        processed_batch_df.to_csv(filename, index=False)
        print(f"Saved batch {batch_num} to {filename}")

    print("\n--- Batch processing finished ---")

# --- Entry point to run the main async function ---
if __name__ == "__main__":
    # This calls asyncio.run() ONLY ONCE to start the entire process
    try:
        asyncio.run(main_batch_processor())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
             print("\nDetected a running event loop. This often happens in interactive environments like Jupyter.")
             print("You might need to run this script directly or use notebook extensions like nest_asyncio.")
             print(f"Original Error: {e}")
        else:
            raise # Re-raise other RuntimeErrors