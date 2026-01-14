import pandas as pd
import numpy as np
import config as cfg
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# ================= CONFIGURATION =================
@dataclass
class InsightConfig:
    """Configuration for insight generation"""
    kpi_list: List[str] = None
    hierarchy_levels: List[str] = None
    partner_column: str = 'Partner Name_x'
    partner_code_column: str = 'Partner Code'
    broker_code_column: str = 'Broker Code'
    relationship_handler_column: str = 'Relationship Handler'
    thresholds: List[int] = None
    top_managers: int = 3
    top_partners: int = 2
    change_threshold: float = 50.0
    year_column: str = 'FY_Year_x'
    
    def __post_init__(self):
        if self.kpi_list is None:
            self.kpi_list = [
                'Equity Sales',
                'SIP Sales_Achievement', 
                'Net Sales through MARS',
                'Investment Net Sales Achievement'
            ]
        if self.hierarchy_levels is None:
            self.hierarchy_levels = ['ZM', 'BM', 'Relationship Handler']  
        if self.thresholds is None:
            self.thresholds = [25, 50, 75, 80, 90]

# ================= GEMINI SERVICE =================
class GeminiService:
    """Handles all Gemini LLM interactions"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(__name__)
    
    def generate_insight(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate insight using Gemini model"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_output_tokens": 800
                }
            )
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"Error generating insight: {e}"

# ================= BASE INSIGHT GENERATOR =================
class InsightGenerator:
    """Base class for all insight generators"""
    def __init__(self, gemini_service: GeminiService, config: InsightConfig):
        self.gemini = gemini_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that required columns exist in dataframe"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing columns in dataframe: {missing_columns}")
            return False
        return True
    
    def get_top_managers(self, df: pd.DataFrame, kpi: str, hierarchy_level: str) -> List[str]:
        """Get top managers by KPI for given hierarchy level"""
        if not self.validate_dataframe(df, [hierarchy_level, kpi]):
            return []
            
        # Filter out NaN values in hierarchy level
        df_clean = df[df[hierarchy_level].notna()]
        
        if df_clean.empty:
            return []
        
        return (
            df_clean.groupby(hierarchy_level)[kpi]
            .sum()
            .sort_values(ascending=False)
            .head(self.config.top_managers)
            .index
            .tolist()
        )

# ================= PARTNER CONCENTRATION =================
class ConcentrationInsightGenerator(InsightGenerator):
    """Generates partner concentration insights"""
    
    def calculate_concentration(self, df: pd.DataFrame, kpi: str) -> Tuple[Dict, int, pd.DataFrame]:
        """Calculate concentration metrics for a group"""
        required_cols = [self.config.partner_column, kpi]
        if not self.validate_dataframe(df, required_cols):
            return {}, 0, pd.DataFrame()
            
        try:
            # Filter out partners with no name
            df_clean = df[df[self.config.partner_column].notna()]
            
            if df_clean.empty:
                return {}, 0, pd.DataFrame()
            
            df_sorted = (
                df_clean.groupby(self.config.partner_column)[kpi]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            total_kpi = df_sorted[kpi].sum()
            if total_kpi == 0:
                return {}, len(df_sorted), df_sorted
                
            df_sorted['Cumulative KPI'] = df_sorted[kpi].cumsum()
            df_sorted['Cumulative KPI %'] = df_sorted['Cumulative KPI'] / total_kpi * 100

            total_partners = len(df_sorted)
            conc_summary = {}

            for threshold in self.config.thresholds:
                try:
                    num_partners = (df_sorted['Cumulative KPI %'] >= threshold).idxmax() + 1
                    conc_summary[f'Partners for {threshold}%'] = num_partners
                    conc_summary[f'% of Total Partners for {threshold}%'] = round(
                        num_partners / total_partners * 100, 2
                    )
                except (ValueError, IndexError):
                    conc_summary[f'Partners for {threshold}%'] = total_partners
                    conc_summary[f'% of Total Partners for {threshold}%'] = 100.0

            return conc_summary, total_partners, df_sorted
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration: {e}")
            return {}, 0, pd.DataFrame()
    
    def generate_manager_concentration(self, df: pd.DataFrame, kpi: str, hierarchy_level: str) -> Dict:
        """Generate concentration insights for all managers at given hierarchy level"""
        required_cols = [hierarchy_level, self.config.partner_column, kpi]
        if not self.validate_dataframe(df, required_cols):
            self.logger.warning(f"Missing required columns for concentration analysis: {required_cols}")
            return {}
            
        manager_insights = {}
        top_managers = self.get_top_managers(df, kpi, hierarchy_level)
        
        if not top_managers:
            self.logger.warning(f"No managers found for {hierarchy_level} level with KPI {kpi}")
            return {}
        
        for manager in top_managers:
            try:
                manager_data = df[df[hierarchy_level] == manager]
                if manager_data.empty:
                    continue
                    
                conc_summary, total_partners, df_sorted = self.calculate_concentration(manager_data, kpi)
                
                if not conc_summary:
                    continue
                    
                # Create concentration text
                insight_text = self._format_concentration_text(
                    manager, kpi, hierarchy_level, total_partners, conc_summary, df_sorted
                )
                
                # Generate LLM insights
                llm_insight = self._generate_concentration_llm_insight(
                    manager, kpi, hierarchy_level, insight_text
                )
                
                manager_insights[manager] = insight_text + "\n\nLLM Insights:\n" + llm_insight
                
            except Exception as e:
                self.logger.error(f"Error processing manager {manager} for concentration: {e}")
                continue
            
        return manager_insights
    
    def _format_concentration_text(self, manager: str, kpi: str, level: str, 
                                 total_partners: int, conc_summary: Dict, df_sorted: pd.DataFrame) -> str:
        """Format concentration analysis into readable text"""
        text = f"{level}: {manager}\nKPI: {kpi}\n"
        text += f"Total Partners under {manager}: {total_partners}\n\n"
        text += f"Partner Concentration Insight for KPI '{kpi}':\n"
        
        for threshold in self.config.thresholds:
            partners_key = f'Partners for {threshold}%'
            percent_key = f'% of Total Partners for {threshold}%'
            
            if partners_key in conc_summary:
                text += (
                    f" - Top {conc_summary[partners_key]} partners "
                    f"({conc_summary[percent_key]}% of total partners) "
                    f"contribute {threshold}% of total {kpi}.\n"
                )
        
        # Add top partners for context
        if not df_sorted.empty:
            top_partners_data = df_sorted.head(self.config.top_partners)[[self.config.partner_column, kpi, 'Cumulative KPI %']]
            text += f"\nTop {self.config.top_partners} Partners (for reference):\n"
            for _, row in top_partners_data.iterrows():
                partner_name = row[self.config.partner_column]
                text += f" - {partner_name}: {row[kpi]:,.2f} ({row['Cumulative KPI %']:.2f}% cumulative)\n"
            
        return text
    
    def _generate_concentration_llm_insight(self, manager: str, kpi: str, level: str, text: str) -> str:
        """Generate LLM insights for concentration analysis"""
        prompt = f"""
        You are a financial performance analyst.
        Below is data for {level} '{manager}' on KPI '{kpi}' showing partner concentration.
        Analyze how concentrated performance is (few vs many partners contributing).
        Write 3–4 crisp insights about:
        1. Level of concentration (high/medium/low),
        2. Implications on business dependency,
        3. Partner development or risk recommendations.

        Data:
        {text}
        """
        return self.gemini.generate_insight(prompt)

# ================= LEADERS & LAGGERS =================
class LeadersLaggersInsightGenerator(InsightGenerator):
    """Generates leaders and laggers insights"""
    
    def generate_manager_leaders_laggers(self, df: pd.DataFrame, kpi: str, hierarchy_level: str) -> Dict:
        """Generate leaders/laggers insights for all managers at given hierarchy level"""
        required_cols = [hierarchy_level, self.config.partner_column, kpi]
        if not self.validate_dataframe(df, required_cols):
            self.logger.warning(f"Missing required columns for leaders/laggers analysis: {required_cols}")
            return {}
            
        manager_insights = {}
        top_managers = self.get_top_managers(df, kpi, hierarchy_level)
        
        if not top_managers:
            self.logger.warning(f"No managers found for {hierarchy_level} level with KPI {kpi}")
            return {}
        
        for manager in top_managers:
            try:
                manager_data = df[df[hierarchy_level] == manager]
                if manager_data.empty:
                    continue
                    
                insight_text = self._analyze_leaders_laggers(manager, kpi, hierarchy_level, manager_data)
                
                # Generate LLM insights
                llm_insight = self._generate_leaders_laggers_llm_insight(
                    manager, kpi, hierarchy_level, insight_text
                )
                
                manager_insights[manager] = insight_text + "\n\nLLM Insights:\n" + llm_insight
                
            except Exception as e:
                self.logger.error(f"Error processing manager {manager} for leaders/laggers: {e}")
                continue
            
        return manager_insights
    
    def _analyze_leaders_laggers(self, manager: str, kpi: str, level: str, df: pd.DataFrame) -> str:
        """Analyze leaders and laggers for a manager"""
        try:
            # Filter out partners with no name
            df_clean = df[df[self.config.partner_column].notna()]
            
            if df_clean.empty:
                return f"{level}: {manager}\nKPI: {kpi}\n\nNo partner data available."
            
            # Sort partners by performance
            perf = (
                df_clean.groupby(self.config.partner_column)[kpi]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            if perf.empty:
                return f"{level}: {manager}\nKPI: {kpi}\n\nNo partner data available."

            total_perf = perf[kpi].sum()
            if total_perf == 0:
                return f"{level}: {manager}\nKPI: {kpi}\n\nNo performance data available."

            # Leaders (Top N)
            leaders = perf.head(self.config.top_partners).copy()
            leaders['% Share'] = round(leaders[kpi] / total_perf * 100, 2)

            # Laggers (Bottom N)
            laggers = perf.tail(self.config.top_partners).copy()
            laggers['% Share'] = round(laggers[kpi] / total_perf * 100, 2)

            # Performance summary
            leaders_contrib = leaders['% Share'].sum()
            laggers_contrib = laggers['% Share'].sum()

            text = f"{level}: {manager}\nKPI: {kpi}\n\n"
            text += f"Total Partners under {manager}: {len(perf)}\n"
            text += f"Top {self.config.top_partners} Leaders contribute: {leaders_contrib:.2f}% of total {kpi}\n"
            text += f"Bottom {self.config.top_partners} Laggers contribute: {laggers_contrib:.2f}% of total {kpi}\n\n"

            text += "Top Performing (Leader) Partners:\n"
            for _, row in leaders.iterrows():
                partner_name = row[self.config.partner_column]
                text += f" - {partner_name}: {row[kpi]:,.2f} ({row['% Share']}%)\n"

            text += "\nLow Performing (Lagger) Partners:\n"
            for _, row in laggers.iterrows():
                partner_name = row[self.config.partner_column]
                text += f" - {partner_name}: {row[kpi]:,.2f} ({row['% Share']}%)\n"

            return text
            
        except Exception as e:
            self.logger.error(f"Error analyzing leaders/laggers for {manager}: {e}")
            return f"{level}: {manager}\nKPI: {kpi}\n\nError analyzing data: {e}"
    
    def _generate_leaders_laggers_llm_insight(self, manager: str, kpi: str, level: str, text: str) -> str:
        """Generate LLM insights for leaders/laggers analysis"""
        prompt = f"""
        You are a financial performance analyst.
        Below is data for {level} '{manager}' on KPI '{kpi}' showing top and bottom performing partners.

        Write a professional 3–4 bullet point insight covering:
        1. Performance dependency (e.g., dominated by few top partners or evenly spread),
        2. Impact of leaders on total performance,
        3. Weak link from laggers and improvement recommendations,
        4. Any early warning or strategic focus points.

        Data:
        {text}
        """
        return self.gemini.generate_insight(prompt)

# ================= MAIN INSIGHT ORCHESTRATOR =================
class InsightOrchestrator:
    """Orchestrates all insight generation across hierarchy levels"""
    
    def __init__(self, gemini_api_key: str, config: InsightConfig = None):
        self.config = config or InsightConfig()
        self.gemini_service = GeminiService(gemini_api_key)
        
        # Initialize all generators
        self.concentration_gen = ConcentrationInsightGenerator(self.gemini_service, self.config)
        self.leaders_laggers_gen = LeadersLaggersInsightGenerator(self.gemini_service, self.config)
        self.drastic_change_gen = DrasticChangeInsightGenerator(self.gemini_service, self.config)
        self.focus_area_gen = FocusAreaInsightGenerator(self.gemini_service, self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def validate_data_requirements(self, df: pd.DataFrame) -> bool:
        """Validate that all required columns exist in the dataframe"""
        required_columns = [self.config.partner_column, self.config.year_column] + self.config.kpi_list
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            return False
            
        return True
    
    def generate_all_insights(self, df: pd.DataFrame) -> Dict:
        """Generate all types of insights across all hierarchy levels"""
        if not self.validate_data_requirements(df):
            return {"error": "Data validation failed - check required columns"}
            
        all_insights = {}
        
        # Process hierarchy levels in the order specified (ZM first)
        for hierarchy_level in self.config.hierarchy_levels:
            if hierarchy_level not in df.columns:
                self.logger.warning(f"Hierarchy level {hierarchy_level} not found in dataframe. Available: {list(df.columns)}")
                continue
                
            level_insights = {}
            
            # Generate insights for each KPI at this hierarchy level
            kpi_insights = {}
            for kpi in self.config.kpi_list:
                if kpi not in df.columns:
                    self.logger.warning(f"KPI {kpi} not found in dataframe")
                    continue
                    
                # 1. Partner Concentration
                try:
                    concentration_insights = self.concentration_gen.generate_manager_concentration(
                        df, kpi, hierarchy_level
                    )
                    if concentration_insights:
                        kpi_insights[kpi] = {
                            'concentration': concentration_insights
                        }
                except Exception as e:
                    self.logger.error(f"Concentration insight error for {kpi}: {e}")
                
                # 2. Leaders & Laggers
                try:
                    leaders_laggers_insights = self.leaders_laggers_gen.generate_manager_leaders_laggers(
                        df, kpi, hierarchy_level
                    )
                    if leaders_laggers_insights:
                        if kpi not in kpi_insights:
                            kpi_insights[kpi] = {}
                        kpi_insights[kpi]['leaders_laggers'] = leaders_laggers_insights
                except Exception as e:
                    self.logger.error(f"Leaders/Laggers insight error for {kpi}: {e}")
            
            if kpi_insights:
                level_insights['kpi_insights'] = kpi_insights
            
            # 3. Drastic Changes (partner level) - only for BM level
            if hierarchy_level == 'BM':
                try:
                    drastic_changes = self.drastic_change_gen.generate_change_insights(df)
                    if drastic_changes:
                        level_insights['drastic_changes'] = drastic_changes
                except Exception as e:
                    self.logger.error(f"Drastic change insight error: {e}")
            
            # 4. Focus Areas (partner level) - only for BM level
            if hierarchy_level == 'BM':
                try:
                    focus_areas = self.focus_area_gen.generate_focus_insights(df)
                    if focus_areas:
                        level_insights['focus_areas'] = focus_areas
                except Exception as e:
                    self.logger.error(f"Focus area insight error: {e}")
            
            if level_insights:
                all_insights[hierarchy_level] = level_insights
            
        return all_insights
    
    def display_insights(self, insights: Dict):
        """Display insights in formatted output"""
        if 'error' in insights:
            print(f"ERROR: {insights['error']}")
            return
            
        # Display insights in the order of hierarchy levels (ZM first)
        for hierarchy_level in self.config.hierarchy_levels:
            if hierarchy_level not in insights:
                continue
                
            level_insights = insights[hierarchy_level]
            print(f"\n{'='*80}")
            print(f"INSIGHTS FOR {hierarchy_level} LEVEL")
            print(f"{'='*80}")
            
            # Display KPI-based insights
            if 'kpi_insights' in level_insights:
                for kpi, insight_types in level_insights['kpi_insights'].items():
                    print(f"\n--- {kpi} ---")
                    for insight_type, manager_insights in insight_types.items():
                        print(f"\n{insight_type.upper()} INSIGHTS:")
                        if isinstance(manager_insights, dict):
                            for manager, insight in manager_insights.items():
                                print(f"\n{manager}:\n{insight}")
                                print("-" * 60)
                        else:
                            print(f"\n{manager_insights}")
            
            # Display other insights
            for insight_type in ['drastic_changes', 'focus_areas']:
                if insight_type in level_insights:
                    print(f"\n{insight_type.upper()} INSIGHTS:")
                    insights_data = level_insights[insight_type]
                    if isinstance(insights_data, dict):
                        for entity, insight in insights_data.items():
                            print(f"\n{entity}:\n{insight}")
                            print("-" * 60)
                    else:
                        print(f"\n{insights_data}")

# ================= USAGE EXAMPLE =================
def main():
    """Example usage of the insight generation system"""
    
    # Configuration matching your actual column names - ZM first
    insight_config = InsightConfig(
        kpi_list=[
            'Equity Sales',
            'SIP Sales_Achievement',
            'Net Sales through MARS',
            'Investment Net Sales Achievement'
        ],
        hierarchy_levels=['ZM', 'BM', 'Relationship Handler'],  # ZM first
        partner_column='Partner Name_x',
        year_column='FY_Year_x',
        top_managers=3,
        top_partners=2,
        change_threshold=50.0
    )
    
    # Initialize orchestrator
    orchestrator = InsightOrchestrator(
        gemini_api_key= 'AIzaSyADwv71iu5J5m9TK0oV2lDgZxQrTLkg1K0',
        config=insight_config
    )
    
    # Generate all insights
    all_insights = orchestrator.generate_all_insights(merged_with_hierarchyy)
    
    # Display insights
    orchestrator.display_insights(all_insights)
    
    return all_insights

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run insight generation
    insights = main()