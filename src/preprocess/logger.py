# --- START OF FILE logger.py ---

import os
import markdown
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, subject_nr, log_base_path):
        self.subject_nr = subject_nr if isinstance(subject_nr, str) else str(subject_nr).zfill(2)
        self.log_path = os.path.join(log_base_path, f"subject_{self.subject_nr}")
        os.makedirs(self.log_path, exist_ok=True)
        self.markdown_file = os.path.join(self.log_path, f"subject_{self.subject_nr}_log.md")
        
        # Open in append mode, create if it doesn't exist.
        # This allows subsequent scripts to add to the same log file.
        if not os.path.exists(self.markdown_file):
            with open(self.markdown_file, 'w') as f:
                f.write(f"# Preprocessing Log for Subject {self.subject_nr}\n\n")

    def log_text(self, text):
        with open(self.markdown_file, 'a') as f:
            f.write(text + "\n\n")

    def log_section(self, section_title):
        with open(self.markdown_file, 'a') as f:
            f.write(f"\n## {section_title}\n\n")

    def log_subsection(self, subsection_title):
        with open(self.markdown_file, 'a') as f:
            f.write(f"### {subsection_title}\n\n")

    def save_plot(self, plt_obj, plot_name):
        plot_filename = f"{plot_name}.png"
        plot_path = os.path.join(self.log_path, plot_filename)
        # Use plt_obj.gcf() if a figure object is passed, otherwise assume it's pyplot
        fig = plt_obj.gcf() if hasattr(plt_obj, 'gcf') else plt_obj
        fig.savefig(plot_path)
        plt.close(fig) # Close the figure to free memory
        with open(self.markdown_file, 'a') as f:
            f.write(f"![{plot_name}]({plot_filename})\n\n")
    
    def convert_markdown_to_html(self):
        with open(self.markdown_file, 'r') as f:
            text = f.read()
        html = markdown.markdown(text, extensions=['extra', 'tables'])
        html_file = os.path.splitext(self.markdown_file)[0] + '.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Converted markdown log to HTML at {html_file}")