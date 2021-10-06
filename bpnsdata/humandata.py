import urllib

import numpy as np
import xarray as xr
import rasterio
from tqdm.auto import tqdm


class HumanData:
    def __init__(self):
        self.resolution = 0.00833333

    def __call__(self, df):
        return self.get_shipping_intensity_df(df)

    def get_shipping_intensity_df(self, df, layer='st'):
        """
        Will return the intensity of the year and month of each sample from EMODnet Human Activities
        https://www.emodnet-humanactivities.eu/

        Parameters
        ----------
        df: DataFrame
            Dataframe to add the shipping intensity to
        layer: str
            Can be 'rd' for Route Density or 'st' for Shipping intensity
        """
        bbox = df.total_bounds
        # Make sure the bbox is big enough to download some data
        if abs(bbox[2] - bbox[0]) <= self.resolution * 2:
            bbox[0] -= 2 * self.resolution
            bbox[2] += 2 * self.resolution
        if abs(bbox[3] - bbox[1]) <= self.resolution * 2:
            bbox[1] -= 2 * self.resolution
            bbox[3] += 2 * self.resolution
        bbox = ",".join(bbox.astype(str))
        df['route_dens'] = np.nan
        for (year, month), df_slice in tqdm(df.groupby([df.index.year, df.index.month])):
            df_slice = df_slice.to_crs(epsg=4326)
            request = 'https://ows.emodnet-humanactivities.eu/wcs?service=wcs&version=1.0.0&request=getcoverage&' \
                      'coverage=emodnet:%s_%02d_%s_All&crs=EPSG:4326&BBOX=%s&' \
                      'format=image/tiff&interpolation=nearest&resx=%s&resy=%s' % \
                      (year, month, layer, bbox, self.resolution, self.resolution)
            try:
                tif_file = urllib.request.urlretrieve(request)
                tif_raster = xr.open_rasterio(tif_file[0])
                for row in df_slice[['geometry']].itertuples():
                    if row[1] is not None:
                        x = row[1].xy[0][0]
                        y = row[1].xy[1][0]
                        df.loc[row.Index, 'route_dens'] = tif_raster.sel(x=x, y=y, method='nearest').values[0]
            except rasterio.errors.RasterioIOError:
                print('Year %s and month %s was not downloaded' % (year, month))
        return df
