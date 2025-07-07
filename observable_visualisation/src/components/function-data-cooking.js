function countFunctionOccurences(acc, curr) {
    // the accumulator is a map of the function's name, and the key the number of time it occurs
    if(curr['qualities_en'] !== undefined && curr['qualities_en'] !== null) {
        for(let i = 0; i < curr["qualities_en"].length; ++i){
            let curr_quality = curr["qualities_en"][i];
            if (acc[curr_quality]) {
                acc[curr_quality] += 1;
            } else {
                acc[curr_quality] = 1;
            }
        }
    }
    return acc;
}

export function cookData(registryData, N) {
    let groupedInstitutions = Object.groupBy(registryData, v => v.owner_standardised);
    let NMostRepresentedInstitutions = Object.entries(groupedInstitutions)
        .map(([key, value]) => {
            return {
                name: key,
                count: value.length,
                qualities: value.length > 0? value.reduce(countFunctionOccurences,{}): {}
            };
        })
        .sort((a, b) => b.count - a.count)
        .filter(v => v.name !== 'possessore ignoto')
        .slice(0, N);
    let vs = NMostRepresentedInstitutions.flatMap(v => {
        return Object.entries(v.qualities).map(k => {
            return {
                "name": v.name,
                "count": k[1],
                "quality": k[0],
            }
        })
    });
    return vs.filter(v => v.count > 2 && v.quality !== "");
}

function buildGeometryIdTotalAreaMap(parcelData) {
    let geometryIdMap = new Map();
    parcelData.features.forEach(entry => {
        if (entry.properties.geometry_id) {
            let geometryId = String(entry.properties.geometry_id);
            if (geometryIdMap.has(geometryId)) {
                geometryIdMap.set(geometryId, geometryIdMap.get(geometryId) + entry.properties.area);
            } else {
                geometryIdMap.set(geometryId, entry.properties.area);
            }
        }
    });
    return geometryIdMap;
}

function countFunctionSurface(acc, curr, geometryIdMap) {
    // the accumulator is a map of the function's name, and the key the number of time it occurs
    if(curr['qualities_en'] !== undefined && curr['qualities_en'] !== null) {
        for(let i = 0; i < curr["qualities_en"].length; ++i){
            let curr_quality = curr["qualities_en"][i];
            if (acc[curr_quality]) {
                acc[curr_quality] += geometryIdMap.get(String(curr.geometry_id)) || 0;
            } else {
                acc[curr_quality] = geometryIdMap.get(String(curr.geometry_id)) || 0;
            }
        }
    }
    return acc;
}

export function cookDataInSurfaceArea(registryData, parcelData, N) {
    let groupedInstitutions = Object.groupBy(registryData, v => v.owner_standardised);
    const geometryIdMap = buildGeometryIdTotalAreaMap(parcelData);
    let NMostRepresentedInstitutions = Object.entries(groupedInstitutions)
        .map(([key, value]) => {
            return {
                name: key,
                count: value.length,
                qualities: value.length > 0? value.reduce((acc, curr) => countFunctionSurface(acc, curr, geometryIdMap),{}): {}
            };
        })
        .sort((a, b) => b.count - a.count)
        .filter(v => v.name !== 'possessore ignoto')
        .slice(0, N);
    let vs = NMostRepresentedInstitutions.flatMap(v => {
        return Object.entries(v.qualities).map(k => {
            return {
                "name": v.name,
                "surface": k[1],
                "quality": k[0],
            }
        })
    });
    return vs.filter(v => v.surface > 0 && v.quality !== "");
}