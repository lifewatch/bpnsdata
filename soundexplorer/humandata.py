import urllib

import numpy as np
import rasterio
import requests
from tqdm.auto import tqdm


class HumanData:
    @staticmethod
    def get_shipping_intensity_df(df, layer='rd'):
        """
        Will return the intensity of the year and month of each sample from EMODnet Human Activities
        https://www.emodnet-humanactivities.eu/

        Parameters
        ----------
        df: DataFrame
            Dataframe to add the shipping intenstiy to
        layer: str
            Can be 'rd' for Route Density or 'st' for Shipping intensity
        """
        bbox = df.total_bounds
        # Make sure the bbox is big enough to download some data
        if abs(bbox[2] - bbox[0]) <= 0.00833333 * 2:
            bbox[0] -= 2 * 0.00833333
            bbox[2] += 2 * 0.00833333
        if abs(bbox[3] - bbox[1]) <= 0.00833333 * 2:
            bbox[1] -= 2 * 0.00833333
            bbox[3] += 2 * 0.00833333
        bbox = ",".join(bbox.astype(str))
        years = df.index.year.unique()
        months = df.index.month.unique()
        df[('route_dens', 'all')] = None
        for year in years:
            for month in months:
                request = 'https://ows.emodnet-humanactivities.eu/wcs?service=wcs&version=1.0.0&request=getcoverage&' \
                          'coverage=emodnet:%s_%02d_%s_All&crs=EPSG:4326&BBOX=%s&' \
                          'format=image/tiff&interpolation=nearest&resx=0.00833333&resy=0.00833333' % \
                          (year, month, layer, bbox)
                response = requests.get(request)
                if response.status_code != 200:
                    print('No shipping density layer found for %s-%s. Rows values set to None' % (year, month))
                else:
                    try:
                        tif_file = urllib.request.urlretrieve(request)
                        tif_raster = rasterio.open(tif_file[0])
                        for idx, row in tqdm(df.iterrows(), total=len(df)):
                            if row['geometry', ''] is not None:
                                if idx.month == month and idx.year == year:
                                    x = row['geometry', ''].xy[0][0]
                                    y = row['geometry', ''].xy[1][0]
                                    row, col = tif_raster.index(x, y)
                                    df.loc[idx, ('route_dens', 'all')] = tif_raster.read(1)[row, col]
                    except rasterio.errors.RasterioIOError:
                        print('Year %s and month %s was not downloaded' % (year, month))
                        idxs = df[(df.index.month == month) & (df.index.year == year)].index
                        df.loc[idxs, ('route_dens', 'all')] = np.nan
        return df
