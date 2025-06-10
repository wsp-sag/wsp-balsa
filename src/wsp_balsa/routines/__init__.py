from .general import (align_categories, is_identifier, reindex_series,
                      sort_nicely, sum_df_sequence)
from .io import *
from .matrices import (aggregate_matrix, disaggregate_matrix, fast_stack,
                       fast_unstack, matrix_balancing_1d, matrix_balancing_2d,
                       matrix_bucket_rounding, split_zone_in_matrix)
from .modelling import distance_array, distance_matrix, tlfd
from .time import generate_time_bin_labels

try:
    from .plotting import (convergence_boxplot, location_summary,
                           trumpet_diagram)
except ImportError:
    pass
