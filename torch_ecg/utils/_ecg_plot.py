"""
BSD 3-Clause License

Copyright (c) 2024, The Alphanumerics Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Modified from https://github.com/alphanumericslab/ecg-image-kit/blob/main/codes/ecg-image-generator/ecg_plot.py
"""

import os
import random
from math import ceil
from random import randint
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from PIL import Image

__all__ = ["ecg_plot"]


standard_values = {
    "y_grid_size": 0.5,
    "x_grid_size": 0.2,
    "y_grid_inch": 5 / 25.4,
    "x_grid_inch": 5 / 25.4,
    "grid_line_width": 0.5,
    "lead_name_offset": 0.5,
    "lead_fontsize": 11.0,
    "x_gap": 1.0,
    "y_gap": 0.5,
    "display_factor": 1.0,
    "line_width": 0.75,
    "row_height": 8.0,
    "dc_offset_length": 0.2,
    "lead_length": 3.0,
    "V1_length": 12.0,
    "width": 11.0,
    "height": 8.5,
}

# standard_major_colors = {
#     "colour1": (1, 0.6823, 0.776),
#     "colour2": (1, 0.796, 0.866),
#     "colour3": (0.933, 0.8235, 0.8),
#     "colour4": (0.996, 0.807, 0.8039),
#     "colour5": (0.611, 0.627, 0.647),
#     "colour6": (0.4901, 0.498, 0.513),
#     "colour7": (0.4274, 0.196, 0.1843),
#     "colour8": (0.992, 0.7529, 0.7254),
#     "colour9": (0.9215, 0.9372, 0.9725),
# }

standard_major_colors = {
    "colour1": (1, 0.6823, 0.776),
    "colour2": (1, 0.796, 0.866),
    "colour3": (0.933, 0.8235, 0.8),
    "colour4": (0.996, 0.807, 0.8039),
    "colour5": (0.4274, 0.196, 0.1843),
    "colour6": (0.992, 0.7529, 0.7254),
}


standard_minor_colors = {
    "colour1": (0.9843, 0.9019, 0.9529),
    "colour2": (0.996, 0.9294, 0.9725),
    "colour3": (0.9529, 0.8745, 0.8549),
    "colour4": (0.996, 0.9529, 0.9529),
    "colour5": (0.5882, 0.4196, 0.3960),
    "colour6": (0.996, 0.8745, 0.8588),
}

# standard_minor_colors = {
#     "colour1": (0.9843, 0.9019, 0.9529),
#     "colour2": (0.996, 0.9294, 0.9725),
#     "colour3": (0.9529, 0.8745, 0.8549),
#     "colour4": (0.996, 0.9529, 0.9529),
#     "colour5": (0.7607, 0.7725, 0.7843),
#     "colour6": (0.6039, 0.6235, 0.6274),
#     "colour7": (0.5882, 0.4196, 0.3960),
#     "colour8": (0.996, 0.8745, 0.8588),
#     "colour9": (0.9568, 0.9686, 0.9843),
# }

papersize_values = {
    "A0": (33.1, 46.8),
    "A1": (33.1, 23.39),
    "A2": (16.54, 23.39),
    "A3": (11.69, 16.54),
    "A4": (8.27, 11.69),
    "letter": (8.5, 11),
}

leadNames_12 = ["III", "aVF", "V3", "V6", "II", "aVL", "V2", "V5", "I", "aVR", "V1", "V4"]


def inches_to_dots(value: float, resolution: int) -> float:
    return value * resolution


def create_signal_dictionary(signal: np.ndarray, full_leads: List[str]) -> Dict[str, np.ndarray]:
    record_dict = {}
    for k in range(len(full_leads)):
        record_dict[full_leads[k]] = signal[k]
    return record_dict


# Function to plot raw ecg signal
def ecg_plot(
    ecg: Dict[str, np.ndarray],
    sample_rate: int,
    columns: int,
    rec_file_name: os.PathLike,
    output_dir: os.PathLike,
    resolution: int = 200,
    pad_inches: int = 0,
    lead_index: Optional[List[str]] = None,
    full_mode: str = "None",
    store_text_bbox: bool = False,
    units: str = "mV",
    papersize: Optional[str] = None,
    x_gap: float = standard_values["x_gap"],
    y_gap: float = standard_values["y_gap"],
    display_factor: float = standard_values["display_factor"],
    line_width: float = standard_values["line_width"],
    title: Optional[str] = None,
    style: Optional[str] = None,
    row_height: float = standard_values["row_height"],
    show_lead_name: bool = True,
    show_grid: bool = False,
    show_dc_pulse: bool = False,
    y_grid: Optional[float] = None,
    x_grid: Optional[float] = None,
    standard_colours: bool = False,
    bbox: bool = False,
    save_format: Optional[str] = None,
) -> Tuple[float, float]:
    """Function to plot ECG signal.

    Parameters
    ----------
    ecg : Dict[str, np.ndarray]
        Dictionary of ECG signals with lead names as keys,
        values as 1D numpy arrays.
    sample_rate : int
        Sampling rate of the ECG signal.
    columns : int
        Number of columns to be plotted in each row.
    rec_file_name : os.PathLike
        Name of the record file.
    output_dir : os.PathLike
        Output directory.
    resolution : int, default ``200``
        Resolution of the output image.
    pad_inches : int, default ``0``
        Padding of white border along the image in inches.
    lead_index : List[str], optional
        Order of lead indices to be plotted.
        By default, the order is the same as the order in ``ecg``.
    full_mode : str, default ``"None"``
        Sets the lead to add at the bottom of the paper ECG as a long strip.
        If ``"None"``, no lead is added at the bottom.
        If not ``"None"``, the lead ``"full" + full_mode`` must be present in ``ecg``.
    store_text_bbox : bool, default ``False``
        If ``True``, stores the bounding box of the text in a text file.
    units : str, default ``"mV"``
        Units of the ECG signal.
    papersize : {``"A0"``, ``"A1"``, ``"A2"``, ``"A3"``, ``"A4"``, ``"letter"``}, default ``None``
        Size of the paper to plot the ECG on.
    x_gap : float, default ``1.0``
        Gap between paper x axis border and signal plot.
    y_gap : float, default ``0.5``
        Gap between paper y axis border and signal plot.
    display_factor : float, default ``1.0``
        Factor to scale the ECG signal by.
    line_width : float, default ``0.75``
        Width of line tracing the ECG.
    title : str, optional
        Title of the figure.
    style : {``"bw"``, ``"color"``}, optional
        Sets the style of the plot.
    row_height : float, default ``8.0``
        Gap between corresponding ECG rows.
    show_lead_name : bool, default ``True``
        Option to show lead names or skip.
    show_grid : bool, default ``False``
        Turn grid on or off.
    show_dc_pulse : bool, default ``False``
        Option to show DC pulse.
    y_grid : float, optional
        Sets the y grid size in inches.
    x_grid : float, optional
        Sets the x grid size in inches.
    standard_colours : bool, default ``False``
        If ``True``, uses standard colours for the plot.
    bbox : bool, default ``False``
        If ``True``, stores the bounding box of the lead in a text file.
    save_format : {``"png"``, ``"pdf"``, ``"svg"``, ``"jpg"``}, optional
        Format to save the plot in.
        If ``None``, the plot is not saved.

    Returns
    -------
    x_grid_dots : float
        X grid size in dots.
    y_grid_dots : float
        Y grid size in dots.

    """

    matplotlib.use("Agg")
    randindex = randint(0, 99)
    random_sampler = random.uniform(-0.05, 0.004)

    # check if the ecg dict is empty
    if ecg == {}:
        return 0, 0

    secs = len(list(ecg.items())[0][1]) / sample_rate

    if lead_index is None:
        lead_index = list(ecg.keys())
    leads = len(lead_index)

    rows = int(ceil(leads / columns))

    if full_mode != "None":
        rows += 1
        leads += 1

    # Grid calibration
    # Each big grid corresponds to 0.2 seconds and 0.5 mV
    # To do: Select grid size in a better way
    y_grid_size = standard_values["y_grid_size"]
    x_grid_size = standard_values["x_grid_size"]
    grid_line_width = standard_values["grid_line_width"]
    lead_name_offset = standard_values["lead_name_offset"]
    lead_fontsize = standard_values["lead_fontsize"]

    # Set max and min coordinates to mark grid. Offset x_max slightly (i.e by 1 column width)

    if not papersize:
        width = standard_values["width"]
        height = standard_values["height"]
    else:
        width = papersize_values[papersize][1]
        height = papersize_values[papersize][0]

    if y_grid is None:
        y_grid = standard_values["y_grid_inch"]
    if x_grid is None:
        x_grid = standard_values["x_grid_inch"]
    y_grid_dots = y_grid * resolution
    x_grid_dots = x_grid * resolution

    # row_height = height * y_grid_size/(y_grid*(rows+2))
    row_height = (height * y_grid_size / y_grid) / (rows + 2)
    x_max = width * x_grid_size / x_grid
    x_min = 0
    x_gap = np.floor(((x_max - (columns * secs)) / 2) / 0.2) * 0.2
    y_min = 0
    y_max = height * y_grid_size / y_grid

    # Set figure and subplot sizes
    fig, ax = plt.subplots(figsize=(width, height))

    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    fig.suptitle(title)

    # Mark grid based on whether we want black and white or colour

    if style == "bw":
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    elif standard_colours:
        random_colour_index = randint(1, 6)
        color_major = standard_major_colors["colour" + str(random_colour_index)]
        color_minor = standard_minor_colors["colour" + str(random_colour_index)]
        randcolorindex_grey = randint(0, 24)
        grey_random_color = random.uniform(0, 0.2)
        color_line = (grey_random_color, grey_random_color, grey_random_color)
    else:
        randcolorindex_red = randint(0, 24)
        major_random_color_sampler_red = random.uniform(0, 0.8)
        randcolorindex_green = randint(0, 24)
        major_random_color_sampler_green = random.uniform(0, 0.5)
        randcolorindex_blue = randint(0, 24)
        major_random_color_sampler_blue = random.uniform(0, 0.5)

        randcolorindex_minor = randint(0, 24)
        minor_offset = random.uniform(0, 0.2)
        minor_random_color_sampler_red = major_random_color_sampler_red + minor_offset
        minor_random_color_sampler_green = random.uniform(0, 0.5) + minor_offset
        minor_random_color_sampler_blue = random.uniform(0, 0.5) + minor_offset

        randcolorindex_grey = randint(0, 24)
        grey_random_color = random.uniform(0, 0.2)
        color_major = (major_random_color_sampler_red, major_random_color_sampler_green, major_random_color_sampler_blue)
        color_minor = (minor_random_color_sampler_red, minor_random_color_sampler_green, minor_random_color_sampler_blue)

        color_line = (grey_random_color, grey_random_color, grey_random_color)

    # Set grid
    # Standard ecg has grid size of 0.5 mV and 0.2 seconds. Set ticks accordingly
    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, x_grid_size))
        ax.set_yticks(np.arange(y_min, y_max, y_grid_size))
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # set grid line style
        ax.grid(which="major", linestyle="-", linewidth=grid_line_width, color=color_major)

        ax.grid(which="minor", linestyle="-", linewidth=grid_line_width, color=color_minor)

    else:
        ax.grid(False)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    lead_num = 0

    # Step size will be number of seconds per sample i.e 1/sampling_rate
    step = 1.0 / sample_rate

    dc_offset = 0
    if show_dc_pulse:
        dc_offset = sample_rate * standard_values["dc_offset_length"] * step
    # Iterate through each lead in lead_index array.
    y_offset = row_height / 2
    x_offset = 0

    text_bbox = []
    lead_bbox = []

    for i in np.arange(len(lead_index)):
        if len(lead_index) == 12:
            leadName = leadNames_12[i]
        else:
            leadName = lead_index[i]
        # y_offset is computed by shifting by a certain offset based on i, and also by row_height/2 to account for half the waveform below the axis
        if i % columns == 0:

            y_offset += row_height

        # x_offset will be distance by which we shift the plot in each iteration
        if columns > 1:
            x_offset = (i % columns) * secs

        else:
            x_offset = 0

        # Create dc pulse wave to plot at the beginning of plot. Dc pulse will be 0.2 seconds
        x_range = np.arange(0, sample_rate * standard_values["dc_offset_length"] * step + 4 * step, step)
        dc_pulse = np.ones(len(x_range))
        dc_pulse = np.concatenate(((0, 0), dc_pulse[2:-2], (0, 0)))

        # Print lead name at .5 ( or 5 mm distance) from plot
        if show_lead_name:
            t1 = ax.text(x_offset + x_gap, y_offset - lead_name_offset - 0.2, leadName, fontsize=lead_fontsize)

            if store_text_bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent()
                x1 = bb.x0 * resolution / fig.dpi
                y1 = bb.y0 * resolution / fig.dpi
                x2 = bb.x1 * resolution / fig.dpi
                y2 = bb.y1 * resolution / fig.dpi
                text_bbox.append([x1, y1, x2, y2, leadName])

        # If we are plotting the first row-1 plots, we plot the dc pulse prior to adding the waveform
        if columns == 1 and i in np.arange(0, rows):
            if show_dc_pulse:
                # Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(x_range + x_offset + x_gap, dc_pulse + y_offset, linewidth=line_width * 1.5, color=color_line)
                if bbox:
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()
                    x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                    x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        elif columns == 4 and i == 0 or i == 4 or i == 8:
            if show_dc_pulse:
                # Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(
                    np.arange(0, sample_rate * standard_values["dc_offset_length"] * step + 4 * step, step) + x_offset + x_gap,
                    dc_pulse + y_offset,
                    linewidth=line_width * 1.5,
                    color=color_line,
                )
                if bbox:
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()
                    x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                    x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        t1 = ax.plot(
            np.arange(0, len(ecg[leadName]) * step, step) + x_offset + dc_offset + x_gap,
            ecg[leadName] + y_offset,
            linewidth=line_width,
            color=color_line,
        )
        if bbox:
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()
            if show_dc_pulse is False or (columns == 4 and (i != 0 and i != 4 and i != 8)):
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi
            else:
                y1 = min(y1, bb.y0 * resolution / fig.dpi)
                y2 = max(y2, bb.y1 * resolution / fig.dpi)
                x2 = bb.x1 * resolution / fig.dpi

            lead_bbox.append([x1, y1, x2, y2, 0])

        start_ind = round((x_offset + dc_offset + x_gap) * x_grid_dots / x_grid_size)
        end_ind = round((x_offset + dc_offset + x_gap + len(ecg[leadName]) * step) * x_grid_dots / x_grid_size)

    # Plotting longest lead for 12 seconds
    if full_mode != "None":
        if show_lead_name:
            t1 = ax.text(x_gap, row_height / 2 - lead_name_offset, full_mode, fontsize=lead_fontsize)

            if store_text_bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent(renderer=fig.canvas.renderer)
                x1 = bb.x0 * resolution / fig.dpi
                y1 = bb.y0 * resolution / fig.dpi
                x2 = bb.x1 * resolution / fig.dpi
                y2 = bb.y1 * resolution / fig.dpi
                text_bbox.append([x1, y1, x2, y2, full_mode])

        if show_dc_pulse:
            t1 = ax.plot(
                x_range + x_gap,
                dc_pulse + row_height / 2 - lead_name_offset + 0.8,
                linewidth=line_width * 1.5,
                color=color_line,
            )

            if bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1[0].get_window_extent()
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        dc_full_lead_offset = 0
        if show_dc_pulse:
            dc_full_lead_offset = sample_rate * standard_values["dc_offset_length"] * step

        t1 = ax.plot(
            np.arange(0, len(ecg["full" + full_mode]) * step, step) + x_gap + dc_full_lead_offset,
            ecg["full" + full_mode] + row_height / 2 - lead_name_offset + 0.8,
            linewidth=line_width,
            color=color_line,
        )

        if bbox:
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()
            if show_dc_pulse is False:
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi
            else:
                y1 = min(y1, bb.y0 * resolution / fig.dpi)
                y2 = max(y2, bb.y1 * resolution / fig.dpi)
                x2 = bb.x1 * resolution / fig.dpi

            lead_bbox.append([x1, y1, x2, y2, 1])

        start_ind = round((dc_full_lead_offset + x_gap) * x_grid_dots / x_grid_size)
        end_ind = round((dc_full_lead_offset + x_gap + len(ecg["full" + full_mode]) * step) * x_grid_dots / x_grid_size)

    head, tail = os.path.split(rec_file_name)
    rec_file_name = os.path.join(output_dir, tail)

    if save_format is not None:
        save_format = f""".{save_format.strip(".")}"""
    else:
        return x_grid_dots, y_grid_dots

    plt.savefig(os.path.join(output_dir, tail + ".png"), dpi=resolution)
    plt.close(fig)
    plt.clf()
    plt.cla()

    if pad_inches != 0:

        ecg_image = Image.open(os.path.join(output_dir, tail + ".png"))

        right = pad_inches * resolution
        left = pad_inches * resolution
        top = pad_inches * resolution
        bottom = pad_inches * resolution
        width, height = ecg_image.size
        new_width = width + right + left
        new_height = height + top + bottom
        result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
        result_image.paste(ecg_image, (left, top))

        result_image.save(os.path.join(output_dir, tail + ".png"))

        plt.close("all")
        plt.close(fig)
        plt.clf()
        plt.cla()

    if store_text_bbox:
        if os.path.exists(os.path.join(output_dir, "text_bounding_box")) is False:
            os.mkdir(os.path.join(output_dir, "text_bounding_box"))

        with open(os.path.join(output_dir, "text_bounding_box", tail + ".txt"), "w") as f:
            for i, l in enumerate(text_bbox):
                if pad_inches != 0:
                    l[0] += left
                    l[2] += left
                    l[1] += top
                    l[3] += top

                for val in l[:4]:
                    f.write(str(val))
                    f.write(",")
                f.write(str(l[4]))
                f.write("\n")

    if bbox:
        if os.path.exists(os.path.join(output_dir, "lead_bounding_box")) is False:
            os.mkdir(os.path.join(output_dir, "lead_bounding_box"))
        with open(os.path.join(output_dir, "lead_bounding_box", tail + ".txt"), "w") as f:
            for i, l in enumerate(lead_bbox):
                if pad_inches != 0:
                    l[0] += left
                    l[2] += left
                    l[1] += top
                    l[3] += top

                for val in l[:4]:
                    f.write(str(val))
                    f.write(",")
                f.write(str(l[4]))
                f.write("\n")

    return x_grid_dots, y_grid_dots
