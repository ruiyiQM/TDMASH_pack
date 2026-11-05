"""Extract unconstrained complex poles and test fixed-nuclei Cayley dispersion."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

import euler_c2h4_cayley as study


def cayley(omega, dt):
    return 2 / dt * np.arctan(np.asarray(omega) * dt / 2)


def pencil_data(time, signal):
    """Factor one multichannel Hankel pencil for all tested truncation ranks."""
    time, signal = np.asarray(time), np.asarray(signal)
    if (time.ndim != 1 or signal.shape != (len(time), 3) or len(time) < 20
            or not np.isfinite(signal).all() or not np.isfinite(time).all()):
        raise ValueError('Harmonic inversion needs at least 20 finite three-component samples.')
    h = time[1] - time[0]
    if h <= 0 or not np.allclose(np.diff(time), h, atol=1e-9, rtol=0):
        raise ValueError('Harmonic inversion requires a uniform increasing time grid.')
    stride = max(1, int(np.floor(.5 / h)))
    # Anti-alias filtering is essential before reducing fine-CN samples.
    y = resample_poly(signal, 1, stride, axis=0, padtype='line') if stride > 1 else signal.copy()
    h *= stride
    t = np.arange(len(y)) * h
    scale = np.linalg.norm(y)
    if scale < 1e-15:
        return dict(scale=0., h=h)
    y = y / scale
    y -= y.mean(axis=0)
    width = min(600, len(y) // 3)
    H0 = np.concatenate([hankel(y[:width, i], y[width - 1:-1, i]) for i in range(3)])
    H1 = np.concatenate([hankel(y[1:width + 1, i], y[width:, i]) for i in range(3)])
    U, values, Vh = svd(H0, full_matrices=False, check_finite=False)
    return dict(h=h, t=t, y=y, scale=scale, U=U, values=values, Vh=Vh, H1=H1)


def fit_pencil(data, rank, cutoff=1e-7):
    if data['scale'] == 0:
        return dict(modes=[], relative_fit_error=0., rank=0, sample_dt=data['h'])
    h, t, y, scale, U, values, Vh, H1 = (data[key] for key in ('h', 't', 'y', 'scale', 'U', 'values', 'Vh', 'H1'))
    rank = min(rank, int(np.count_nonzero(values > cutoff * values[0])))
    poles = np.linalg.eigvals((U[:, :rank].conj().T @ H1 @ Vh[:rank].conj().T) / values[:rank])
    exponents = np.log(poles.astype(complex)) / h
    keep = np.isfinite(exponents) & (abs(exponents.real) * t[-1] < 30)
    exponents = exponents[keep]
    basis = np.exp(np.outer(t, exponents))
    basis = np.c_[np.ones(len(t)), t / max(t[-1], 1), basis]
    coefficients, _, _, _ = np.linalg.lstsq(basis, y, rcond=1e-10)
    error = float(np.linalg.norm(basis @ coefficients - y) / max(np.linalg.norm(y), 1e-30))
    modes = []
    for exponent, coefficient in zip(exponents, coefficients[2:]):
        if exponent.imag <= .005:
            continue
        amplitude = coefficient * scale
        modes.append(dict(omega=float(exponent.imag), growth_rate=float(exponent.real),
                          amplitude=float(2 * np.linalg.norm(amplitude)),
                          polarization_real=amplitude.real.tolist(), polarization_imag=amplitude.imag.tolist()))
    modes.sort(key=lambda mode: mode['amplitude'], reverse=True)
    return dict(modes=modes, relative_fit_error=error, rank=rank, sample_dt=h)


def harmonic_inversion(time, signal, rank=320, cutoff=1e-7):
    """Fit multichannel exponential modes without using predicted frequencies."""
    return fit_pencil(pencil_data(time, signal), rank, cutoff)


def extract(time, signal):
    pencil = pencil_data(time, signal)
    fits = [fit_pencil(pencil, rank) for rank in (320, 352, 384)]
    primary = fits[2]
    for mode in primary['modes']:
        differences = []
        for fit in (fits[0], fits[1]):
            if fit['modes']:
                differences.append(min(abs(other['omega'] - mode['omega']) for other in fit['modes']))
        mode['rank_spread_au'] = max(differences, default=None)
    if len(time) >= 40:
        shortened = harmonic_inversion(time[:int(.85 * len(time))], signal[:int(.85 * len(time))], rank=384)
        for mode in primary['modes']:
            mode['window_spread_au'] = min((abs(other['omega'] - mode['omega']) for other in shortened['modes']), default=None)
    return primary


def load(root, case, kick):
    folder = root / 'runs' / f"case_{case['case_id']:03d}" / f'kick_{kick}'
    record = json.loads((folder / 'result.json').read_text())
    failure = record.get('failure')
    if failure and 'did not converge' in failure['message'] and failure['iterations'] != study.ITERATIONS:
        # Frozen PT records can retain the last accepted step's iteration counter.
        failure['previous_step_iterations'] = failure['iterations']
        failure['iterations'] = study.ITERATIONS
    raw = folder / 'trajectory.npz'
    if study.digest(raw) != record['trajectory_sha256']:
        raise RuntimeError('Trajectory checksum mismatch: ' + str(raw))
    with np.load(raw, allow_pickle=False) as data:
        return record, {name: data[name].copy() for name in
                        ('time', 'dipole', 'diagnostic_time', 'density_displacement', 'energy')}


def linear_window(time, small, large):
    """Retain a contiguous response prefix; do not fit a nonlinear tail as a spectrum."""
    block = max(8, round(60 / (time[1] - time[0])))
    end, worst, history = len(time), 0., []
    for stop in range(block, len(time) + block, block):
        stop = min(stop, len(time))
        first = max(0, stop - block)
        difference = np.linalg.norm(small[first:stop] - large[first:stop])
        magnitude = .5 * (np.linalg.norm(small[first:stop]) + np.linalg.norm(large[first:stop]))
        error = float(difference / max(magnitude, 1e-20))
        history.append(dict(time_au=float(time[stop - 1]), normalized_response_disagreement=error))
        worst = max(worst, error)
        if error > .05:
            end = first
            break
    return max(end, 1), worst, history


def match_modes(reference, candidate, dt):
    """Report assignments, ambiguity and residuals rather than replacing measured poles."""
    targets = reference[:5]
    options = candidate
    if not targets or not options:
        return []
    predictions = cayley([mode['omega'] for mode in targets], dt)
    cost = abs(predictions[:, None] - np.array([mode['omega'] for mode in options])[None, :])
    rows, columns = linear_sum_assignment(cost)
    results = []
    for i, j in zip(rows, columns):
        source, measured = targets[i], options[j]
        sorted_cost = np.sort(cost[i])
        spread = max(measured.get('rank_spread_au') or 0., measured.get('window_spread_au') or 0.)
        ref_spread = max(source.get('rank_spread_au') or 0., source.get('window_spread_au') or 0.,
                         source.get('reference_refinement_error_au') or 0.)
        predicted_shift = float(predictions[i] - source['omega'])
        results.append(dict(omega_ref=source['omega'], omega_meas=measured['omega'],
                            omega_cayley=float(predictions[i]), mapping_error=float(measured['omega'] - predictions[i]),
                            predicted_shift=predicted_shift, measured_shift=measured['omega'] - source['omega'],
                            shift_resolvable=bool(abs(predicted_shift) > max(3 * spread, 3 * ref_spread, 5e-6)),
                            amplitude_ratio=measured['amplitude'] / source['amplitude'],
                            growth_rate=measured['growth_rate'], fit_spread_au=spread, reference_spread_au=ref_spread,
                            nearest_pole_assignment=True,
                            assignment_margin_au=float(sorted_cost[1] - sorted_cost[0]) if len(sorted_cost) > 1 else None,
                            resolved=bool(spread < .002 and abs(measured['omega'] - predictions[i]) < .02)))
    return results


def spectrum(time, signal, omega):
    """Use a common physical damping only for plotting, never for pole locations."""
    h = time[1] - time[0]
    stride = max(1, int(np.floor(.5 / h)))
    y = resample_poly(signal, 1, stride, axis=0, padtype='line') if stride > 1 else signal
    t = np.arange(len(y)) * h * stride
    y = (y - y.mean(axis=0)) * np.exp(-t[:, None] / 300.)
    result = np.zeros(len(omega))
    for start in range(0, len(omega), 100):
        transform = np.exp(1j * np.outer(omega[start:start + 100], t)) @ y * h * stride
        result[start:start + 100] = np.linalg.norm(transform, axis=1)
    return result


def analyze(root):
    study.verify(root)
    calibration_path = root / 'harmonic_calibration_v4.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    configuration = json.loads((root / 'cases.json').read_text())
    records, signals = [], {}
    for case in configuration:
        loaded = [load(root, case, kick) for kick in range(3)]
        count = min(len(data['time']) for _, data in loaded)
        t = loaded[0][1]['time'][:count]
        if not all(np.allclose(data['time'][:count], t, rtol=0, atol=1e-9) for _, data in loaded):
            raise ValueError('Kick and baseline must share physical nodes.')
        small, large = [(loaded[i][1]['dipole'][:count] - loaded[0][1]['dipole'][:count]) / study.KICKS[i] for i in (1, 2)]
        end, disagreement, history = linear_window(t, small, large) if count > 1 else (1, None, [])
        complete = all(record['completed'] and np.isclose(record['final_time_au'], study.DURATION) for record, _ in loaded)
        physical = json.loads((root / ('response_' + case['state'] + '.json')).read_text())
        fit = extract(t[:end], small[:end]) if end >= 40 and t[end - 1] >= 30 else None
        row = dict(case, completed=complete, linear_response_until_au=float(t[end - 1]),
                   full_window_linear=bool(end == count and complete), kick_disagreement_max=disagreement,
                   physical_response_unstable=physical['unstable'], linearity_windows=history,
                   propagation_records=[record for record, _ in loaded], fit=fit)
        if physical['unstable']:
            row['classification'] = 'unstable_mean_field_response; no stationary-spectrum verdict'
            unstable = next(mode for mode in physical['growing_modes'] if mode['omega'] > 0)
            eigenvalue = complex(unstable['omega'], unstable['growth_rate'])
            factor = (1 - .5j * case['dt'] * eigenvalue) / (1 + .5j * case['dt'] * eigenvalue)
            predicted = dict(omega=float(-np.angle(factor) / case['dt']),
                             growth_rate=float(np.log(abs(factor)) / case['dt']))
            observed = min(fit['modes'], key=lambda mode: abs(mode['omega'] - predicted['omega'])
                           + abs(mode['growth_rate'] - predicted['growth_rate'])) if fit and fit['modes'] else None
            row['complex_cayley_check'] = dict(prediction=predicted, observed=observed,
                                               prediction_for='PT linearized trapezoid; a hypothesis for conventional orbital CN')
        elif not complete:
            row['classification'] = 'solver_or_endpoint_failure; incomplete spectrum'
        elif end < count:
            row['classification'] = 'nonlinear_kick_response; no full-window linear spectrum'
        elif fit is None:
            row['classification'] = 'insufficient_linear_time_data'
        if end > 2:
            signals[case['case_id']] = (t[:end], small[:end])
        records.append(row)
    reference_checks = {}
    for state in ('ground', 'dscf'):
        coarse, fine = [row for row in records if row['state'] == state and row['role'] == 'reference']
        checks = []
        if coarse['fit'] and fine['fit']:
            checks = match_modes(fine['fit']['modes'], coarse['fit']['modes'], .01)
            for mode in fine['fit']['modes']:
                check = next((item for item in checks if abs(item['omega_ref'] - mode['omega']) < 1e-10), None)
                if check is not None:
                    mode['reference_refinement_error_au'] = abs(check['omega_meas'] - check['omega_ref']) / 3
        valid = bool(checks and coarse['full_window_linear'] and fine['full_window_linear']
                     and coarse['fit']['relative_fit_error'] < .05 and fine['fit']['relative_fit_error'] < .05
                     and all(abs(r['omega_meas'] - r['omega_ref']) < .001 for r in checks))
        reference_checks[state] = dict(validated=valid, pole_comparisons=checks,
                                       caveat='Only stable linear full-window references can certify the long-time mapping.')
        for row in records:
            if row['state'] != state or not row['fit'] or not fine['fit']:
                continue
            row['peak_comparisons'] = match_modes(fine['fit']['modes'], row['fit']['modes'], row['dt'])
            for peak in row['peak_comparisons']:
                peak['frequency_measurement_resolved'] = peak['fit_spread_au'] < .002
                margin = peak['assignment_margin_au']
                if margin is not None:
                    peak['frequency_measurement_resolved'] &= margin > max(3 * peak['fit_spread_au'], 1e-5)
                peak['calibration_error_au'] = 0.
                if row['method'] == 'pt_cn' and calibration:
                    known = next(item for item in calibration['records'] if abs(item['dt'] - row['dt']) < 1e-10)
                    control = min(known['bright_peak_checks'], key=lambda item: abs(item['known_omega'] - peak['omega_cayley']))
                    peak['known_pole_calibration'] = control
                    peak['calibration_error_au'] = control['absolute_error']
                    peak['frequency_measurement_resolved'] &= control['well_resolved']
            usable = [peak for peak in row['peak_comparisons'] if peak['frequency_measurement_resolved'] and peak['shift_resolvable']]
            row['resolved_mode_count'] = len(usable)
            if row['physical_response_unstable'] or not row['full_window_linear']:
                row['classification'] = 'unstable_or_nonlinear; no stationary-spectrum verdict'
            elif not valid or row['fit']['relative_fit_error'] > .05:
                row['classification'] = 'unresolved; reference or harmonic fit not certified'
            elif not usable:
                row['classification'] = 'shift_or_mode_unresolved'
            elif all(abs(p['mapping_error']) <= max(1e-5, 3 * (p['fit_spread_au']**2 + p['reference_spread_au']**2 + p['calibration_error_au']**2)**.5) for p in usable):
                row['classification'] = 'mapping_supported_for_matched_resolved_modes'
            else:
                row['classification'] = 'mapping_not_supported_for_resolved_modes'
    paired = []
    for state in ('ground', 'dscf'):
        for dt in study.TIMESTEPS:
            pair = [next(row for row in records if row['state'] == state and row['dt'] == dt and row['method'] == method)
                    for method in ('accelerated', 'pt_cn')]
            comparisons = [row.get('peak_comparisons', []) for row in pair]
            paired.append(dict(state=state, dt=dt, both_completed=all(row['completed'] for row in pair),
                               stable_linear=all(row['full_window_linear'] and not row['physical_response_unstable'] for row in pair),
                               peaks=[dict(omega_ref=left['omega_ref'], omega_cn=left['omega_meas'], omega_pt=right['omega_meas'],
                                           pt_minus_cn=right['omega_meas'] - left['omega_meas'],
                                           amplitude_pt_over_cn=right['amplitude_ratio'] / left['amplitude_ratio'])
                                      for left in comparisons[0] for right in comparisons[1]
                                      if abs(left['omega_ref'] - right['omega_ref']) < 1e-10]))
    summary = dict(protocol=json.loads((root / 'protocol.json').read_text()),
                   analysis_script_sha256=study.digest(Path(__file__)),
                   reference_checks=reference_checks, records=records,
                   same_dt_cn_pt_comparisons=paired,
                   largest_tested_completed_dt={state: {method: max((row['dt'] for row in records
                       if row['state'] == state and row['method'] == method and row['role'] == 'candidate' and row['completed']), default=None)
                       for method in ('accelerated', 'pt_cn')} for state in ('ground', 'dscf')},
                   response={state: json.loads((root / ('response_' + state + '.json')).read_text()) for state in ('ground', 'dscf')},
                   caveats=['No unconditional equality of CN and PT-CN spectra is assumed.',
                            'Finite-window harmonic inversion does not resolve every crowded or dark mode.',
                            'Nearest-pole matching uses theory only after unconstrained pole extraction; ambiguous matches are recorded.'])
    if calibration:
        summary['known_pole_calibration'] = calibration
    study.write_json(root / 'compact_summary.json', summary)
    plot(root, records, signals)


def plot(root, rows, signals):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={'height_ratios': [1.3, 1, .75]})
    colors = plt.get_cmap('viridis')(np.linspace(.05, .95, len(study.TIMESTEPS)))
    for column, state in enumerate(('ground', 'dscf')):
        upper, lower, amplitude = axes[:, column]
        selected = [row for row in rows if row['state'] == state]
        finest = next(row for row in selected if row['role'] == 'reference' and row['dt'] == .005)
        grid = np.linspace(.01, 1.5, 1600)
        if finest['case_id'] in signals:
            t, y = signals[finest['case_id']]
            value = spectrum(t, y, grid)
            upper.plot(grid, value, color='black', lw=1.8, label='CN reference 0.005')
        for row in selected:
            if row['role'] != 'candidate':
                continue
            color = colors[study.TIMESTEPS.index(row['dt'])]
            style = '-' if row['method'] == 'pt_cn' else '--'
            marker = 'o' if row['method'] == 'pt_cn' else 'x'
            if row['case_id'] in signals:
                t, y = signals[row['case_id']]
                available = grid < np.pi / row['dt']
                value = spectrum(t, y, grid[available])
                upper.plot(grid[available], value, color=color, ls=style, lw=.8, alpha=.85,
                           label=f"{'PT' if row['method'] == 'pt_cn' else 'CN'} {row['dt']:g}")
            comparisons = row.get('peak_comparisons', [])
            if comparisons:
                for peak in comparisons:
                    resolved = peak.get('frequency_measurement_resolved', False) and peak['shift_resolvable']
                    point_color, point_marker = (color, marker) if resolved else ('#aaaaaa', '+')
                    lower.scatter(peak['omega_ref'], peak['omega_meas'], color=point_color, marker=point_marker, s=23)
                    uncertainty = 3 * (peak['fit_spread_au']**2 + peak.get('calibration_error_au', 0)**2)**.5
                    lower.errorbar(peak['omega_ref'], peak['omega_meas'], yerr=uncertainty, color=point_color, alpha=.45, lw=.7)
                    amplitude.scatter(peak['omega_ref'], peak['amplitude_ratio'], color=point_color, marker=point_marker, s=23)
            if row['method'] == 'pt_cn':
                x = np.linspace(.02, 1.5, 300)
                lower.plot(x, cayley(x, row['dt']), color=color, lw=.8, alpha=.7)
        if finest['physical_response_unstable']:
            upper.text(.03, .96, 'Unstable response: only linear prefixes shown\nNo long-time stationary spectrum claim',
                       transform=upper.transAxes, va='top', color='#a33', fontsize=9)
        upper.set_title('UKS ground state' if state == 'ground' else r'UKS $\Delta$SCF state')
        upper.set_xlabel(r'Frequency $\omega$ (Ha / $\hbar$)')
        upper.set_ylabel(r'$|\Delta\mu(\omega)|/\kappa$ (common damping)')
        upper.legend(ncol=3, fontsize=6.8, loc='upper right')
        lower.plot([0, 1.5], [0, 1.5], ':', color='gray', lw=1)
        lower.set(xlabel=r'Reference $\omega_{ref}$', ylabel=r'Measured $\omega_{meas}$', xlim=(0, 1.5), ylim=(0, 1.5))
        lower.text(.03, .96, 'Curves: Cayley prediction\no: PT-CN   x: CN   gray +: unresolved', transform=lower.transAxes, va='top', fontsize=8)
        amplitude.axhline(1., color='gray', ls=':', lw=1)
        amplitude.set(xlabel=r'Reference $\omega_{ref}$', ylabel='Fitted amplitude / reference', xlim=(0, 1.5))
        for axis in (upper, lower, amplitude):
            axis.grid(alpha=.2)
    fig.suptitle(r'C$_2$H$_4$ fixed nuclei: unconstrained poles vs Cayley mapping, $\alpha=1$', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(root / 'cayley_spectra.png', dpi=160)
    plt.close(fig)


def self_check():
    t = np.arange(0, 1200. + 2., 2.)
    omega, gamma = .313271, .0007
    y = np.sin(omega * t)[:, None] * np.exp(gamma * t[:, None]) * np.array([[1., .4, -.2]])
    fit = harmonic_inversion(t, y, rank=12)
    mode = min(fit['modes'], key=lambda x: abs(x['omega'] - omega))
    assert abs(mode['omega'] - omega) < 1e-7 and abs(mode['growth_rate'] - gamma) < 1e-7
    h = 5.
    prediction = cayley(omega, h)
    z = (1 + .5j * h * omega) / (1 - .5j * h * omega)
    assert abs(np.angle(z) / h - prediction) < 1e-14 and abs(abs(z) - 1) < 1e-14
    assert prediction < omega and abs(mode['amplitude'] - np.linalg.norm([1., .4, -.2])) < 1e-5
    close = (.3, .3011)
    signal = (np.sin(close[0] * t)[:, None] * np.array([[1., .3, .1]])
              + .8 * np.sin(close[1] * t)[:, None] * np.array([[.2, 1., -.4]]))
    signal += np.random.default_rng(20261104).normal(scale=1e-7, size=signal.shape)
    fitted = harmonic_inversion(t, signal, rank=8)
    assert all(min(abs(mode['omega'] - w) for mode in fitted['modes']) < 1e-5 for w in close)
    end, _, _ = linear_window(t, y, y * 1.001)
    assert end == len(t)
    changed = y.copy()
    changed[t > 300] *= 2
    end, _, _ = linear_window(t, y, changed)
    assert 240 < t[end - 1] < 360
    try:
        harmonic_inversion(t[:10], y[:10])
    except ValueError:
        pass
    else:
        raise AssertionError('An insufficient time series must not produce a certified pole.')
    print('Sub-bin harmonic poles, growth rates, amplitudes, Cayley phase and nonlinear-prefix checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is not None:
        analyze(args.root)
    else:
        parser.error('--root or --self-check is required')
