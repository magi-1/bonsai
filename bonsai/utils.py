import pandas as pd
import plotly.graph_objects as go

def parallel_coordinates(opt_results : pd.DataFrame):  
  target = opt_results['target']  
  fig = go.Figure(
      data = go.Parcoords(
        line = dict(
            color = target, colorscale = 'RdBu', 
            cmin = target.min(), cmax = target.max()),
        dimensions = [dict(
            label = col, values = opt_results[col],
            range = [opt_results[col].min(), opt_results[col].max()]) 
              for col in opt_results.columns])) ; fig.show()