from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse import optical_assemblies as oa
from pyLaserPulse import single_plot_window
import pyLaserPulse.base_components as bc
import pyLaserPulse.data.paths as paths
import pyLaserPulse.catalogue_components.active_fibres as af
import pyLaserPulse.catalogue_components.fibre_components as fc
import pyLaserPulse.catalogue_components.passive_fibres as pf

import scipy.constants as const


class Yb_fibre_Fabry_Perot:
    def __init__(self, round_trips, round_trip_output_samples=10,
                 high_res_sampling=None, high_res_sampling_limits=None,
                 data_directory=None, plot=False):
        """
        Class for simulating a Fabry-Perot Yb-doped fibre mode-locked laser.

        PM cavity design with a saturable Bragg reflector and a grating
        compressor.

        A ring cavity can be simulated in a similar way.
        """
        self.g = grid.grid(2**11, 1050e-9, 1200e-9)

        # 1 fJ start pulse
        self.p = pulse.pulse(
            1.587e-12, [.592e-3, .592e-3], "Gauss", 40e6, self.g)
        self.tol = 1e-5

        self.round_trips = round_trips
        self.round_trip_output_samples = round_trip_output_samples
        self.high_res_sampling = high_res_sampling
        self.high_res_sampling_limits = high_res_sampling_limits
        self.data_directory = data_directory
        self.plot = plot

    def simulate(
            self, L_gain, L_wdm, L_oc, OC, L_sbr, sbr_loss, sbr_mod_depth,
            sbr_Fsat, sbr_tau, grating_separation, grating_angle, pump_power):

        # GAIN FIBRE
        # The repetition rate can change with cavity fibre lengths given above.
        # The repetition rate must be reset for the gain fibre once it is
        # known.
        # The pump power is set once the repetition rate is known using the
        # gain_fibre.add_pump method.
        pump_points = 2**9
        ASE_wl_lims = [975e-9, 977e-9]
        bounds = {
            'co_pump_power': 0, 'co_pump_wavelength': 976e-9,
            'co_pump_bandwidth': 1e-9}
        time_domain_gain = False
        gain1 = af.Nufern_PM_YSF_HI_HP(
            self.g, L_gain, self.p.repetition_rate, pump_points, ASE_wl_lims,
            bounds, time_domain_gain=time_domain_gain)
        gain2 = af.Nufern_PM_YSF_HI_HP(
            self.g, L_gain, self.p.repetition_rate, pump_points, ASE_wl_lims,
            bounds, time_domain_gain=time_domain_gain)

        # WDM
        # wdm = fc.Opneti_high_power_filter_WDM_1020_1080(self.g, L_wdm, L_wdm)
        wdm_in_fibre = pf.PM980_XP(self.g, L_wdm, self.tol)
        wdm_out_fibre = pf.PM980_XP(self.g, L_wdm, self.tol)
        wdm = bc.fibre_component(
            self.g, wdm_in_fibre, wdm_out_fibre, 0.166, 20e-9, self.g.lambda_c,
            1, 0, 1, 1e-3, output_coupler=False)

        # OUTPUT COUPLER -- first pass
        oc_loss = 0.166
        oc_in_fibre = pf.PM980_XP(self.g, L_oc, self.tol)
        oc_out_fibre = pf.PM980_XP(self.g, L_oc, self.tol)
        oc1 = bc.fibre_component(
            self.g, oc_in_fibre, oc_out_fibre, oc_loss, 40e-9, self.g.lambda_c,
            # 2e-2, 0, 1, 1e-3, output_coupler=False)
            2e-2, 0, OC, 1e-3, output_coupler=True, coupler_type="beamsplitter")

        # OUTPUT COUPLER -- second pass
        oc_loss = 1 - (1 - OC)*(1 - 0.166)  # loss without output coupling
        oc2 = bc.fibre_component(
            self.g, oc_in_fibre, oc_out_fibre, oc_loss, 40e-9, self.g.lambda_c,
            2e-2, 0, 1, 1e-3, output_coupler=False)
            # 2e-2, 0, OC, 1e-3, output_coupler=True, coupler_type="beamsplitter")

        # SBR
        sbr_pigtail = pf.PM980_XP(self.g, L_sbr, self.tol)
        sbr = bc.saturable_Bragg_reflector(
            sbr_loss, sbr_mod_depth, sbr_tau, sbr_Fsat,
            sbr_pigtail.signal_mode_area, self.g)

        # DISPERSION COMPENSATION
        comp = bc.grating_compressor(
            0.1, 200e-9, paths.materials.reflectivities.gold, self.g.lambda_c,
            3e-3, 0, 1, 0, grating_separation, grating_angle, 600, self.g,
            optimize=False, verbose=False, output_coupler=False)

        # APPROXIMATE FREE-SPACE PATH LENGTH (e.g., between a fibre collimator
        # and the dispersion compensation). Say 5cm from the collimator to the
        # first grating, and 5 cm from the second grating the retroreflector.
        single_pass_compressor = 5e-2 + comp.grating_separation + 5e-2

        # The fibre lengths and free-space distance need converting to a
        # repetition rate so that the pump power can be converted to pump
        # energy.
        rep_rate \
            = const.c / (
                gain1.L * gain1.signal_ref_index[self.g.midpoint]
                + wdm.input_fibre.L * wdm.input_fibre.signal_ref_index[self.g.midpoint]
                + wdm.output_fibre.L * wdm.output_fibre.signal_ref_index[self.g.midpoint]
                + oc1.input_fibre.L * oc1.input_fibre.signal_ref_index[self.g.midpoint]
                + oc1.output_fibre.L * oc1.output_fibre.signal_ref_index[self.g.midpoint]
                + sbr_pigtail.L * sbr_pigtail.signal_ref_index[self.g.midpoint]
                + single_pass_compressor * 1.0003
            )
        rep_rate /= 2  # Fabry Perot cavity; optical length is 2x calculated

        self.p.change_repetition_rate(self.g, rep_rate)

        # Pump energy per HALF round trip (the gain is propagated twice).
        # It is extremely difficult to find mode-locking solutions when using
        # a full gain model (i.e., with boundaries solved). This is not due to
        # a longer simulation time, but instead because of the additional
        # constraints on possible solutions. Additionally, little benefit to
        # doing this has been found so far.
        #
        # With this in mind, although in this cavity design the gain fibre would
        # be co-pumped on one pass and counter-pumped on the other, using just
        # the 'simple' gain model and co-pumping in both propagation directions
        # is sufficient.
        gain1.add_pump(976e-9, 1e-9, pump_power/2, rep_rate, direction="co")
        gain1.pump.change_repetition_rate(self.g, self.p.repetition_rate)
        gain2.add_pump(976e-9, 1e-9, pump_power/2, rep_rate, direction="co")

        # components list -- near symmetrical; linear Fabry Perot cavity. Each
        # component is passed twice - once per propagation direction per round
        # trip - with the exception of the cavity ends (SBR and compressor).
        self.component_list = [gain1, wdm, oc1, sbr_pigtail, sbr, sbr_pigtail,
                               oc2, wdm, gain2, comp]

        self.osc = oa.sm_fibre_laser(
            self.g, self.component_list, self.round_trips,
            name="Fabry-Perot oscillator",
            verbose=False, plot=self.plot,
            round_trip_output_samples=self.round_trip_output_samples,
            high_res_sampling=self.high_res_sampling,
            high_res_sampling_limits=self.high_res_sampling_limits,
            data_directory=self.data_directory)
        self.p = self.osc.simulate(self.p)
        return self.p


if __name__ == "__main__":
    laser = Yb_fibre_Fabry_Perot(150, round_trip_output_samples=150,
                                 high_res_sampling=10,
                                 high_res_sampling_limits=[0, 150],
                                 data_directory='/home/james/Desktop/TEST',
                                 plot=True)
    L_gain = 0.55
    L_wdm = 0.64
    L_oc = 0.13
    OC = 0.5
    L_sbr = 0.11
    sbr_loss = 0.22
    mod_depth = 0.34
    Fsat = 0.7
    tau = 700e-15
    grating_sep = 0.083
    grating_angle = 0.32
    pump_power = 0.11

    p = laser.simulate(
        L_gain, L_wdm, L_oc, OC, L_sbr, sbr_loss, mod_depth, Fsat, tau,
        grating_sep, grating_angle, pump_power)

    import pyLaserPulse.single_plot_window as spw
    if laser.plot:
        plot_dicts = [laser.osc.plot_dict]
        spw.matplotlib_gallery.launch_plot(plot_dicts=plot_dicts)

