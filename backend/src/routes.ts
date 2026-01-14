router.get('/ticker/history', async (req, res) => {
    const { universe, symbol } = req.query;
    try {
        const history = await getTickerHistory(universe as string, symbol as string);
        res.json(history);
    } catch (e: any) {
        res.status(500).json({ error: e.message });
    }
});

router.get('/sector/analysis', async (req, res) => {
    const { universe } = req.query;
    try {
        const sectors = await getSectorAnalysis(universe as string);
        res.json(sectors);
    } catch (e: any) {
        res.status(500).json({ error: e.message });
    }
});

export default router;
