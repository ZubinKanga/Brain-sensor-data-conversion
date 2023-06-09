/*
    Copyright (C) 2010-2012  EPFL (Ecole Polytechnique Fédérale de Lausanne)
    Laboratory CNBI (Chair in Non-Invasive Brain-Machine Interface)
    Nicolas Bourdaud <nicolas.bourdaud@epfl.ch>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef EEGDEV_PLUGINAPI_H
#define EEGDEV_PLUGINAPI_H
 
#include <stdbool.h>
#include <stdint.h>

#include "eegdev.h"

#define EEGDEV_PLUGIN_ABI_VERSION  8 //last: plugincap w/ multiple chmap


#ifdef __cplusplus
extern "C" {
#endif

union gval {
	float valfloat;
	double valdouble;
	int32_t valint32_t;
};

struct selected_channels {
	union gval sc; /* to be set if bsc != 0 */
	unsigned int in_offset;
	unsigned int inlen;
	unsigned int typein, typeout;
	unsigned int iarray;
	unsigned int arr_offset;
	int bsc;
	int padding;
};

struct egdi_signal_info {
	const char *unit, *transducer, *prefiltering;
	int isint, bsc, dtype, mmtype;
	double scale;
	union gval min, max;
};


struct egdi_chinfo {
	const char *label;
	const struct egdi_signal_info* si;
	int stype;
};


/* EGDCAP_NOCP_*: use pointer directly (do not copy data) */
#define EGDCAP_NOCP_CHMAP	0x00000001
#define EGDCAP_NOCP_DEVID	0x00000002
#define EGDCAP_NOCP_DEVTYPE	0x00000004
#define EGDCAP_NOCP_CHLABEL	0x00000008

struct blockmapping {
	int nch;
	int num_skipped;
	int skipped_stype;
	const struct egdi_chinfo* chmap;
	const struct egdi_signal_info* default_info;
};

struct plugincap {
	unsigned int sampling_freq;
	int flags;
	unsigned int num_mappings;
	const struct blockmapping* mappings;
	const char* device_type;
	const char* device_id;
};

struct devmodule;

struct core_interface {
/* \param dev		pointer to the devmodule struct of the device
 * \param in		pointer to an array of samples
 * \param length	size in bytes of the array
 *
 * egd_update_ringbuffer() should be called by the device implementation
 * whenever a new piece of data is available. This function updates the
 * ringbuffer with the data pointed by the pointer in. The array can be
 * incomplete, i.e. it can start and end at a position not corresponding to
 * a boundary of a samples. */
	int (*update_ringbuffer)(struct devmodule* dev,
	                                   const void* in, size_t len);


/* \param dev		pointer to the devmodule struct of the device
 * \param num_ingrp	number of channels group sent to the ringbuffer
 *
 * Specifies the number of channels groups the device implementation is
 * going to send to the ringbuffer. 
 *
 * This function returns the allocated in case of success, NULL otherwise.
 *
 * IMPORTANT: This function SHOULD be called by the device implementation
 * while executing set_channel_groups mthod.
 */
	struct selected_channels* (*alloc_input_groups)(
	                                         struct devmodule* dev,
                                                unsigned int num_ingrp);

	void (*report_error)(struct devmodule* dev, int error);

	int (*get_stype)(const char* name);


/* \param dev		pointer to the devmodule struct of the device
 * \param len		size in bytes of one sample
 *
 * Specifies the size of one sample as it is supplied to the function
 * egd_update_ringbuffer. 
 *
 * IMPORTANT: This function SHOULD be called by the device implementation
 * before the first call to egd_update_ringbuffer and before the method
 * set_channel_groups returns. */
	void (*set_input_samlen)(struct devmodule* dev, unsigned int len);

	int (*set_cap)(struct devmodule* dev, const struct plugincap* cap);


/* \param dev		pointer to the devmodule struct of the device
 * \param name		name of the mapping
 * \param nch		pointer to an int value that will receive the number
 *                      of channel in the mapping.
 *
 * Returns the array of channels defining the channel mapping
 *
 * IMPORTANT: This function can be called only while opening the device. */
	const struct egdi_chinfo* (*get_conf_mapping)(struct devmodule* dev,
	                                        const char* name, int* nch);
};

struct egdi_optname {
	const char *name, *defvalue;
};

struct egdi_plugin_info {
	unsigned int plugin_abi;
	unsigned int struct_size;
	int (*open_device)(struct devmodule*, const char*[]);
	int (*close_device)(struct devmodule*);
	int (*set_channel_groups)(struct devmodule*, unsigned int,
	                                            const struct grpconf*);
	int (*start_acq)(struct devmodule*);
	int (*stop_acq)(struct devmodule*);
	void (*fill_chinfo)(const struct devmodule*, int, unsigned int,
	                    struct egdi_chinfo*, struct egdi_signal_info*);
	const struct egdi_optname* supported_opts;
};


struct devmodule {
	const struct core_interface ci;
};


static inline
unsigned int egd_get_data_size(unsigned int type)
{
	unsigned int size = 0;

	if (type == EGD_INT32)		
		size = sizeof(int32_t);
	else if (type == EGD_FLOAT)
		size = sizeof(float);
	else if (type == EGD_DOUBLE)
		size = sizeof(double);
	
	return size;
}

static inline
void egdi_set_gval(union gval* dst, int type, double val)
{
	if (type == EGD_INT32)
		dst->valint32_t = (int32_t) val;
	else if (type == EGD_FLOAT)
		dst->valfloat = (float) val;
	else if (type == EGD_DOUBLE)
		dst->valdouble = val;
}

#ifdef __cplusplus
}
#endif


#endif /* EEGDEV_PLUGINAPI_H */
