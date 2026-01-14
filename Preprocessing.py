import pandas as pd
import numpy as np
import config
import logging
import sys
from datetime import datetime
import os

# Initialize logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log filename with timestamp
    log_filename = f"logs/data_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

google_api_key = config.google_api_key

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.7)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataProcessor initialized")
    
    def load_and_preprocess_business_numbers(self, file_paths):
        """Load and preprocess business numbers files following exact original steps"""
        self.logger.info("Starting Business Numbers data processing")
        
        try:
            self.logger.info(f"Loading Business Numbers files: {file_paths}")
            fy23_BN = pd.read_excel(file_paths[0])
            fy24_BN = pd.read_excel(file_paths[1])
            fy25_BN = pd.read_excel(file_paths[2])
            self.logger.info("All Business Numbers files loaded successfully")

            # Make first row the header - FY23
            self.logger.info("Processing FY23 Business Numbers")
            fy23_BN.columns = fy23_BN.iloc[0]  
            fy23_BN = fy23_BN[1:]               
            fy23_BN = fy23_BN.reset_index(drop=True) 
            self.logger.info(f"FY23 BN processed - shape: {fy23_BN.shape}")

            # Make first row the header - FY24
            self.logger.info("Processing FY24 Business Numbers")
            fy24_BN.columns = fy24_BN.iloc[0]  
            fy24_BN = fy24_BN[1:]              
            fy24_BN = fy24_BN.reset_index(drop=True) 
            self.logger.info(f"FY24 BN processed - shape: {fy24_BN.shape}")

            # Make first row the header - FY25
            self.logger.info("Processing FY25 Business Numbers")
            fy25_BN.columns = fy25_BN.iloc[0]  
            fy25_BN = fy25_BN[1:]              
            fy25_BN = fy25_BN.reset_index(drop=True) 
            self.logger.info(f"FY25 BN processed - shape: {fy25_BN.shape}")

            # Add FY Year columns
            fy23_BN['FY_Year'] = '2023'
            fy24_BN['FY_Year'] = '2024'
            fy25_BN['FY_Year'] = '2025'
            self.logger.info("FY Year columns added")

            BN_total = pd.concat([fy23_BN, fy24_BN, fy25_BN], axis=0)
            self.logger.info(f"Business Numbers concatenated - final shape: {BN_total.shape}")
            self.logger.info(f"BN columns: {list(BN_total.columns)}")
            
            return BN_total
            
        except Exception as e:
            self.logger.error(f"Error processing Business Numbers: {e}", exc_info=True)
            raise
    
    def load_and_preprocess_scorecards(self, file_paths):
        """Load and preprocess scorecard files following exact original steps"""
        self.logger.info("Starting Scorecard data processing")
        
        try:
            self.logger.info(f"Loading Scorecard files: {file_paths}")
            fy23_SC = pd.read_excel(file_paths[0])
            fy24_SC = pd.read_excel(file_paths[1])
            fy25_SC = pd.read_excel(file_paths[2])
            self.logger.info("All Scorecard files loaded successfully")

            # Process FY23_SC - exact same steps as original
            self.logger.info("Processing FY23 Scorecard")
            # Make first row the header
            fy23_SC.columns = fy23_SC.iloc[0]   # assign first row as header
            fy23_SC = fy23_SC[1:]               # drop the first row
            fy23_SC = fy23_SC.reset_index(drop=True) 
            self.logger.info(f"FY23 SC after first header - shape: {fy23_SC.shape}")

            # If your columns are like a MultiIndex (from reading an Excel with merged headers)
            # First, fill forward the top-level header
            cols = fy23_SC.columns.to_list()
            new_cols = []
            last_val = None
            for c in cols:
                if pd.notna(c):
                    last_val = c
                    new_cols.append(c)
                else:
                    new_cols.append(last_val)
            fy23_SC.columns = new_cols
            self.logger.info(f"FY23 SC after fill forward - shape: {fy23_SC.shape}")

            # Now handle the 2nd row if it contains sub-columns like Target/Achievement/% Achievement
            # Suppose the 0th row in your df contains sub-column info
            sub_cols = fy23_SC.iloc[0].fillna('')
            fy23_SC = fy23_SC[1:]  # Remove the 0th row used for sub-headers

            # Combine top-level + sub-column names
            fy23_SC.columns = [f"{top}_{sub}" if sub != '' else top for top, sub in zip(fy23_SC.columns, sub_cols)]

            # Reset index
            fy23_SC.reset_index(drop=True, inplace=True)
            self.logger.info(f"FY23 SC fully processed - shape: {fy23_SC.shape}")

            # Process FY24_SC - exact same steps as original
            self.logger.info("Processing FY24 Scorecard")
            # Make first row the header
            fy24_SC.columns = fy24_SC.iloc[0]   # assign first row as header
            fy24_SC = fy24_SC[1:]               # drop the first row
            fy24_SC = fy24_SC.reset_index(drop=True) 
            self.logger.info(f"FY24 SC after first header - shape: {fy24_SC.shape}")

            # If your columns are like a MultiIndex (from reading an Excel with merged headers)
            # First, fill forward the top-level header
            cols = fy24_SC.columns.to_list()
            new_cols = []
            last_val = None
            for c in cols:
                if pd.notna(c):
                    last_val = c
                    new_cols.append(c)
                else:
                    new_cols.append(last_val)
            fy24_SC.columns = new_cols
            self.logger.info(f"FY24 SC after fill forward - shape: {fy24_SC.shape}")

            # Now handle the 2nd row if it contains sub-columns like Target/Achievement/% Achievement
            # Suppose the 0th row in your df contains sub-column info
            sub_cols = fy24_SC.iloc[0].fillna('')
            fy24_SC = fy24_SC[1:]  # Remove the 0th row used for sub-headers

            # Combine top-level + sub-column names
            fy24_SC.columns = [f"{top}_{sub}" if sub != '' else top for top, sub in zip(fy24_SC.columns, sub_cols)]

            # Reset index
            fy24_SC.reset_index(drop=True, inplace=True)
            self.logger.info(f"FY24 SC fully processed - shape: {fy24_SC.shape}")

            # Process FY25_SC - exact same steps as original
            self.logger.info("Processing FY25 Scorecard")
            fy25_SC.columns = fy25_SC.iloc[0]   
            fy25_SC = fy25_SC[1:]               
            fy25_SC = fy25_SC.reset_index(drop=True) 
            self.logger.info(f"FY25 SC after first header - shape: {fy25_SC.shape}")

            cols = fy25_SC.columns.to_list()
            new_cols = []
            last_val = None
            for c in cols:
                if pd.notna(c):
                    last_val = c
                    new_cols.append(c)
                else:
                    new_cols.append(last_val)
            fy25_SC.columns = new_cols
            self.logger.info(f"FY25 SC after fill forward - shape: {fy25_SC.shape}")

            sub_cols = fy25_SC.iloc[0].fillna('')
            fy25_SC = fy25_SC[1:]  # Remove the 0th row used for sub-headers

            fy25_SC.columns = [f"{top}_{sub}" if sub != '' else top for top, sub in zip(fy25_SC.columns, sub_cols)]

            fy25_SC.reset_index(drop=True, inplace=True)
            self.logger.info(f"FY25 SC fully processed - shape: {fy25_SC.shape}")

            # Add FY Year and concatenate
            fy23_SC['FY_Year'] = '2023'
            fy24_SC['FY_Year'] = '2024'
            fy25_SC['FY_Year'] = '2025'
            self.logger.info("FY Year columns added to Scorecards")

            SC_totals = pd.concat([fy23_SC, fy24_SC, fy25_SC], axis=0)
            self.logger.info(f"Scorecards concatenated - final shape: {SC_totals.shape}")
            self.logger.info(f"SC columns: {list(SC_totals.columns)}")
            
            return SC_totals
            
        except Exception as e:
            self.logger.error(f"Error processing Scorecards: {e}", exc_info=True)
            raise
    
    def load_and_preprocess_mis_data(self, file_paths):
        """Load and preprocess MIS data files following exact original steps"""
        self.logger.info("Starting MIS data processing")
        
        try:
            self.logger.info(f"Loading MIS files: {file_paths}")
            fy24_MIS = pd.read_excel(file_paths[0])
            fy25_MIS = pd.read_excel(file_paths[1])
            self.logger.info("All MIS files loaded successfully")

            # Process FY24_MIS - exact same steps as original
            self.logger.info("Processing FY24 MIS")
            fy24_MIS.columns = fy24_MIS.iloc[0]
            fy24_MIS = fy24_MIS.drop(fy24_MIS.index[0])
            fy24_MIS = fy24_MIS.reset_index(drop=True)
            self.logger.info(f"FY24 MIS processed - shape: {fy24_MIS.shape}")
            
            # Process FY25_MIS - exact same steps as original
            self.logger.info("Processing FY25 MIS")
            fy25_MIS.columns = fy25_MIS.iloc[0]
            fy25_MIS = fy25_MIS.drop(fy25_MIS.index[0])
            fy25_MIS = fy25_MIS.reset_index(drop=True)
            self.logger.info(f"FY25 MIS processed - shape: {fy25_MIS.shape}")

            fy24_MIS['FY_Year'] = '2024'
            fy25_MIS['FY_Year'] = '2025'
            self.logger.info("FY Year columns added to MIS data")

            MIS_total = pd.concat([fy24_MIS, fy25_MIS], axis=0)
            self.logger.info(f"MIS data concatenated - final shape: {MIS_total.shape}")
            self.logger.info(f"MIS columns: {list(MIS_total.columns)}")
            
            return MIS_total
            
        except Exception as e:
            self.logger.error(f"Error processing MIS data: {e}", exc_info=True)
            raise
    
    def merge_all_data(self, bn_total, sc_totals, mis_total):
        """Merge all datasets following exact original steps"""
        self.logger.info("Starting data merging process")
        
        try:
            # Merge Business Numbers and Scorecards - exact same steps as original
            self.logger.info("Merging Business Numbers and Scorecards")
            self.logger.info(f"BN shape before merge: {bn_total.shape}")
            self.logger.info(f"SC shape before merge: {sc_totals.shape}")
            
            merged_df = pd.merge(
                bn_total,
                sc_totals,
                how='left',
                left_on=['Partner Code', 'Partner Name', 'FY_Year'],
                right_on=['Broker Code', 'Partner Name', 'FY_Year'],
                suffixes=('_x', '')
            )
            self.logger.info(f"After BN-SC merge - shape: {merged_df.shape}")

            merged_df = merged_df[[col for col in merged_df.columns if '_x' not in col]]
            self.logger.info(f"After removing _x columns - shape: {merged_df.shape}")

            merged_df = merged_df[merged_df['FY_Year'].isin(['2023','2024','2025'])]
            self.logger.info(f"After year filtering - shape: {merged_df.shape}")

            # Clean data types - exact same steps as original
            self.logger.info("Cleaning data types")
            merged_df['Partner Code'] = merged_df['Partner Code'].astype(str).str.strip()
            merged_df['FY_Year'] = merged_df['FY_Year'].astype(str).str.strip()

            # Merge with MIS data - exact same steps as original
            self.logger.info("Merging with MIS data")
            self.logger.info(f"Merged DF shape before MIS merge: {merged_df.shape}")
            self.logger.info(f"MIS shape before merge: {mis_total.shape}")
            
            all_df = pd.merge(
                merged_df,
                mis_total,
                on='Broker Code',  
                how='left'
            )
            self.logger.info(f"Final merged dataset shape: {all_df.shape}")
            self.logger.info("Data merging completed successfully")
            
            return all_df
            
        except Exception as e:
            self.logger.error(f"Error during data merging: {e}", exc_info=True)
            raise

def main():
    processor = DataProcessor()
    
    # File paths
    business_numbers_files = [
        "FY23 - YTD - Partner Score Card - Business Numbers.xls",
        "FY24 - YTD - Partner Score Card - Business Numbers.xls", 
        "FY25 - YTD - Partner Score Card - Business Numbers.xls"
    ]
    
    scorecard_files = [
        "FY23 - YTD - Partner Score Card - Scores.xls",
        "FY24 - YTD - Partner Score Card - Scores.xls",
        "FY25 - YTD - Partner Score Card - Scores.xls"
    ]
    
    mis_files = [
        "FY24 - Partner MIS Data.xlsx",
        "FY25 - Partner MIS Data.xlsx"
    ]
    
    try:
        logger.info("=== STARTING DATA PROCESSING PIPELINE ===")
        
        print("Loading Business Numbers data...")
        logger.info("Loading Business Numbers data...")
        bn_total = processor.load_and_preprocess_business_numbers(business_numbers_files)
        print(f"BN Total shape: {bn_total.shape}")
        
        print("Loading Scorecard data...")
        logger.info("Loading Scorecard data...")
        sc_totals = processor.load_and_preprocess_scorecards(scorecard_files)
        print(f"SC Totals shape: {sc_totals.shape}")
        
        print("Loading MIS data...")
        logger.info("Loading MIS data...")
        mis_total = processor.load_and_preprocess_mis_data(mis_files)
        print(f"MIS Total shape: {mis_total.shape}")
        
        print("Merging datasets...")
        logger.info("Merging datasets...")
        final_df = processor.merge_all_data(bn_total, sc_totals, mis_total)
        print(f"Final dataset shape: {final_df.shape}")
        
        print("Data processing completed successfully!")
        logger.info("=== DATA PROCESSING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final dataset statistics - Shape: {final_df.shape}, Columns: {len(final_df.columns)}")
        
        return final_df
        
    except Exception as e:
        error_msg = f"Error during data processing: {e}"
        print(error_msg)
        logger.error(error_msg, exc_info=True)
        logger.error("=== DATA PROCESSING FAILED ===")
        return None

# Alternative: Individual file processing functions for debugging
def debug_scorecard_file(file_path, year):
    """Debug individual scorecard file processing"""
    logger.info(f"Debugging scorecard file: {file_path} for year: {year}")
    
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Make first row the header
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index(drop=True)
        logger.info(f"After first header processing - shape: {df.shape}")
        logger.info(f"Columns after first header: {list(df.columns)}")

        # Fill forward the top-level header
        cols = df.columns.to_list()
        new_cols = []
        last_val = None
        for c in cols:
            if pd.notna(c):
                last_val = c
                new_cols.append(c)
            else:
                new_cols.append(last_val)
        df.columns = new_cols
        logger.info(f"After fill forward - shape: {df.shape}")
        logger.info(f"Columns after fill forward: {list(df.columns)}")

        # Handle sub-headers
        sub_cols = df.iloc[0].fillna('')
        df = df[1:]
        df.columns = [f"{top}_{sub}" if sub != '' else top for top, sub in zip(df.columns, sub_cols)]
        df = df.reset_index(drop=True)
        
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Final columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Error debugging file {file_path}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Script started")
    final_data = main()
    
    if final_data is not None:
        logger.info("Script completed successfully")
    else:
        logger.error("Script completed with errors")