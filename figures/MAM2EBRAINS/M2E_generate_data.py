from M2E_compute_pop_rates import compute_pop_rates
from M2E_compute_pop_LvR import compute_pop_LvR
from M2E_compute_corrcoeff import compute_corrcoeff
from M2E_compute_rate_time_series import compute_rate_time_series


def generate_data(M, data_path, label, raster_areas):
    # Compute pop_rates
    compute_pop_rates(M, data_path, label)
    
    # Compute pop_LvR
    compute_pop_LvR(M, data_path, label)
    
    # compute correlation_coefficient
    compute_corrcoeff(M, data_path, label)
    
    # compute rate_time_series_full
    for area in raster_areas:
        compute_rate_time_series(M, data_path, label, area, 'full')
    
    # compute rate_time_series_auto_kernel
    for area in raster_areas:
        compute_rate_time_series(M, data_path, label, area, 'auto_kernel')