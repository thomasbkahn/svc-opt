import warnings
from math import fabs
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import ConvergenceWarning
from tqdm import tqdm
from bokeh.plotting import figure, show, save, output_notebook, output_file, ColumnDataSource, gridplot
from bokeh.models import HoverTool, PrintfTickFormatter
import matplotlib.pyplot as plt
from utilities import cm_to_bokeh, get_color_index

# supress harmless sklearn warnings
warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.")
warnings.simplefilter("ignore", ConvergenceWarning)

class BaseScanner(object):


    def __init__(self, X, y, class_names, n_steps, n_iters, seed):
       
        self.X = X
        self.y = y
        self.n_classes = y.max()+1
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = ['Class '+str(n) for n in range(self.n_classes)]
        self.n_steps = n_steps
        self.n_iters = n_iters
        self.titles = ['Overall Accuracy', 'Training Accuracy', 'Difference']
        self.titles.extend('{name} (F1 Score)'.format(name=name) for name in self.class_names)
        self.tol = 1e-3
        self.max_solver_iters = 500
        self.plot_params = dict()
        self.restore_plot_defaults()
        self._get_seeds(seed)

    def restore_plot_defaults(self):
        
        self.plot_params.update({
            'highlight_max' : 'cornflowerblue',
            'highlight_hovertext' : 'firebrick',
        })

        self.set_cmap(plt.cm.magma)

    def set_cmap(self, cm):
        self.plot_params['cmap'] = cm_to_bokeh(cm)

    def _get_seeds(self, seed):
        randgen = np.random.RandomState(seed)
        self.seeds = randgen.randint(0, int(1e6), int(1e3)).tolist()




class RBFScanner(BaseScanner):


    def __init__(self, X, y, C_lims=(-12,12), gamma_lims=(-12,12), n_steps=50, n_iters=20, logvals=True, class_names=None, seed=None):

        BaseScanner.__init__(self, X, y, class_names, n_steps, n_iters, seed)
        self.clf = SVC(kernel='rbf', decision_function_shape='ovr', max_iter=self.max_solver_iters)
        self.scan(logvals, C_lims, gamma_lims)


    def scan(self, logvals=True, C_lims=None, gamma_lims=None):

        self.logvals = logvals

        if C_lims:
            self.C_vals = np.linspace(C_lims[0], C_lims[1], self.n_steps)
            if self.logvals:
                self.C_vals = 10**self.C_vals

        if gamma_lims:
            self.gamma_vals = np.linspace(gamma_lims[0], gamma_lims[1], self.n_steps)
            if self.logvals:
                self.gamma_vals = 10**self.gamma_vals

        if self.logvals:
            x_vals = np.log10(self.gamma_vals)
            y_vals = np.log10(self.C_vals)
            y_label = 'Log C'
            x_label = 'Log Gamma'
        else:
            x_vals = self.gamma_vals
            y_vals = self.C_vals
            y_label = 'C'
            x_label = 'Gamma'

        self.plot_params.update({
            'x_vals'  : x_vals,
            'y_vals'  : y_vals,
            'x_label' : x_label,
            'y_label' : y_label,
        })

        self.accs = np.zeros((self.n_steps, self.n_steps, self.n_classes+3))

        scan_seed = self.seeds.pop()
        randgen = np.random.RandomState(seed=scan_seed)
        split_seeds = randgen.randint(0, int(1e6), int(self.n_iters * (self.n_steps ** 2))).tolist()
    
        for j, C_val_j in tqdm(enumerate(self.C_vals), total=self.n_steps, desc='Scanning progress'):
            for i, gamma_val_i in enumerate(self.gamma_vals):
                scores = []
                scores_train = []
                f1_arr      = np.zeros((self.n_iters, self.n_classes), dtype=np.float64)
                support_arr = np.zeros((self.n_iters, self.n_classes), dtype=np.int8)
                for n in range(self.n_iters):
                    split_seed = split_seeds.pop()
                    self.clf.set_params(C=C_val_j, gamma=gamma_val_i)
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=split_seed)
                    self.clf.fit(X_train, y_train)
                    y_pred = self.clf.predict(X_test)
                    matches = y_pred == y_test
                    scores.append(matches.sum() / matches.size)
                    f1s, support = precision_recall_fscore_support(y_test, y_pred)[2:]
                    f1_arr[n,:] = f1s
                    support_arr[n,:] = support
                    y_pred = self.clf.predict(X_train)
                    matches = y_pred == y_train
                    scores_train.append(matches.sum() / matches.size)

                f1_avg = (f1_arr * support_arr).sum(axis=0) / support_arr.sum(axis=0)
                self.accs[i,j,3:] = f1_avg
                scores = np.array(scores)
                scores_train = np.array(scores_train)
                self.accs[i,j,0] = scores.mean()
                self.accs[i,j,1] = scores_train.mean()

        self.accs[:,:,2] = self.accs[:,:,1] - self.accs[:,:,0]
        self._find_optimals()


    def show_train_test(self, n_cols=3, dims=(300,300), v_lims=(0.0, 1.0), save_string=None):
        
        plot_titles = [
            'Overall Accuracy',
            'Training Accuracy',
            'Difference'
        ]

        self.plot_params.update({
            'plot_titles' : plot_titles,
            'n_cols'      : n_cols,
            'dims'        : dims,
            'v_lims'      : v_lims,
            'save_string' : save_string
        })

        fig = self._make_multiplot()
        return fig


    def show_classes(self, n_cols=2, dims=(400,400), v_lims=(0.0, 1.0), save_string=None):
        
        plot_titles = ['Overall Accuracy']
        plot_titles.extend(title for title in self.titles if title.endswith('(F1 Score)'))

        self.plot_params.update({
            'plot_titles' : plot_titles,
            'n_cols'      : n_cols,
            'dims'        : dims,
            'v_lims'      : v_lims,
            'save_string' : save_string
        })

        fig = self._make_multiplot()
        return fig


    def show_single(self, title, dims=(500,500), v_lims=(0.0, 1.0), save_string=None):

        self.plot_params.update({
            'plot_titles' : [title],
            'n_cols'      : 1,
            'dims'        : dims,
            'v_lims'      : v_lims,
            'save_string' : save_string
        })

        fig = self._make_multiplot()
        return fig


    def _make_multiplot(self):

        self._make_datasource()


        self._make_hovertool_string()
        figs = [self._make_patch_plot(title) for title in self.plot_params['plot_titles']]

        for fig_i in figs[1:]:
            fig_i.x_range = figs[0].x_range
            fig_i.y_range = figs[0].y_range

        for i, fig in enumerate(figs):
            fig.title_text_color = 'black'
            fig.axis.axis_label_text_color = 'black'
            fig.axis.major_label_text_color = '#B3B3B3'
            fig.title_text_font_size = '18pt'
            fig.axis.axis_label_text_font_size = '12pt'
            fig.axis.major_label_text_font_size= '9pt'
            fig.axis.minor_tick_line_color = None
            fig.axis.major_tick_in = -2
            fig.axis.major_tick_out = 8
            fig.axis.major_tick_line_color = '#B3B3B3'
            fig.axis.major_tick_line_width = 2
            fig.axis.major_tick_line_cap = 'butt'
            fig.xaxis.axis_label = self.plot_params['x_label']
            fig.yaxis.axis_label = self.plot_params['y_label']
            fig.outline_line_width = 0.5
            fig.outline_line_color = 'black'
            if not self.logvals:
                fig.xaxis[0].formatter = PrintfTickFormatter(format="%0.1e")
                fig.yaxis[0].formatter = PrintfTickFormatter(format="%0.1e")
            hover = fig.select_one(HoverTool)
            hover.point_policy = "follow_mouse"
            hover.tooltips = self._hovertool_html[i]

        n_cols = self.plot_params['n_cols']
        n_figs = len(figs)
        figs = [figs[i*n_cols:(i+1)*n_cols] for i in range((n_figs//n_cols)+1)]
        if n_figs % n_cols == 0:
            figs=figs[:-1]
        fig = gridplot(figs)

        save_string = self.plot_params['save_string']

        if not save_string:
            output_notebook()
            show(fig)
        elif save_string == 'return':
            return(fig)
        elif save_string.endswith('.html'):
            output_file(save_string)
            save(fig)


    def _make_datasource(self):

        x_tags     = []
        y_tags     = []
        patches_xs = []
        patches_ys = []
        vals       = [[] for t in self.titles]
        colors     = [[] for t in self.titles]

        x_vals = self.plot_params['x_vals']
        y_vals = self.plot_params['y_vals']

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        delta_x = (x_max - x_min) / (self.n_steps - 1)
        delta_y = (y_max - y_min) / (self.n_steps - 1)

        v_min, v_max = self.plot_params['v_lims']

        for i, xi in enumerate(x_vals):
            for j, yj in enumerate(y_vals):
                x_tags.append('{:.3e}'.format(10**xi if self.logvals else xi))
                y_tags.append('{:.3e}'.format(10**yj if self.logvals else yj))
                patches_xs.append([xi - delta_x/2, xi+delta_x/2, xi+delta_x/2, xi-delta_x/2])
                patches_ys.append([yj - delta_y/2, yj-delta_y/2, yj+delta_y/2, yj+delta_y/2])
                for k, t in enumerate(self.titles):
                    val = self.accs[i,j,k]
                    vals[k].append('{:.3f}'.format(val))
                    if (fabs(val - self.optimal_params[k]['max']) < self.tol) and self.plot_params['highlight_max']:
                        colors[k].append(self.plot_params['highlight_max'])
                    else:
                        colors[k].append(self.plot_params['cmap'][get_color_index(val, v_min, v_max)])
           
        ds_dict = {
            'patch_x' : patches_xs,
            'patch_y' : patches_ys,
            'x_tag' : x_tags,
            'y_tag' : y_tags,
        }

        ds_dict.update({'value'+str(i) : val for i, val in enumerate(vals)})
        ds_dict.update({'color'+str(i) : col for i, col in enumerate(colors)})

        self.datasource = ColumnDataSource(data=ds_dict)


    def _make_patch_plot(self, title):
        
        color_idx = 'color'+str(self.titles.index(title))

        w, h = self.plot_params['dims']
        x_vals = self.plot_params['x_vals']
        y_vals = self.plot_params['y_vals']

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        delta_x = (x_max - x_min) / (self.n_steps - 1)
        delta_y = (y_max - y_min) / (self.n_steps - 1)

        tools = ['hover', 'box_zoom', 'reset']

        f = figure(width=w, height=h, tools=tools, 
                   title=title, x_range=[x_min-delta_x/2, x_max+delta_x/2],
                   y_range=[y_min-delta_y/2, y_max+delta_y/2],)

        f.patches('patch_x', 'patch_y', fill_color=color_idx, 
                  line_color='black', fill_alpha=1.0, line_alpha=1.0, 
                  line_width=0.5, source=self.datasource)

        return f


    def _make_hovertool_string(self):

        head = """
        <div style="font-size: 12px; line-height: 125%;color: #000000">
        <span>Gamma = @x_tag</span>&nbsp;
        <br>
        <span>C = @y_tag</span>&nbsp;
        <hr style="padding:1px;margin-top:4px;margin-bottom:4px" />
        """

        tail = '</div>'

        title_L = self.plot_params['plot_titles']
        value_L = ['value'+str(self.titles.index(title)) for title in title_L]

        self._hovertool_html = []

        for key_title in title_L:
            inner = []
            for title_i, value_i, in zip(title_L, value_L):
                if self.plot_params['highlight_hovertext'] and (title_i == key_title):
                    text_color = self.plot_params['highlight_hovertext']
                else:
                    text_color = '#000000'
                inner.append('<span style="color:{color}">{title} = @{val}</span>&nbsp;\n'.format(color=text_color, title=title_i, val=value_i))

            inner = '<br>\n'.join(inner)
            self._hovertool_html.append(head+inner+tail)


    def _find_optimals(self):

        self.optimal_params = []

        n_rows = self.C_vals.size
        n_cols = self.gamma_vals.size

        for i, t in enumerate(self.titles):
            arr = self.accs[:,:,i]
            max_idx   = arr.argmax()
            row_idx   = max_idx // n_rows
            col_idx   = max_idx % n_cols
            max_val   = arr[row_idx, col_idx]
            max_C     = self.C_vals[row_idx]
            max_gamma = self.gamma_vals[col_idx]
            n_degen   = ((np.abs(arr - max_val)) < self.tol).sum()
            self.optimal_params.append({
                    'title'      : t,
                    'max'        : max_val,
                    'C'          : max_C,
                    'gamma'      : max_gamma,
                    'degeneracy' : n_degen,
                })
